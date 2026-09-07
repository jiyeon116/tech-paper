"""Tests for explicit recurrent replay and independent specialist learners."""

from __future__ import annotations

from types import SimpleNamespace
import os
import unittest
from unittest import mock

import numpy as np

from dynamic_experiment import learner


class _FakeNetwork:
    def __init__(self, scope: str, identifier: int, n_actions: int):
        self.scope = scope
        self.n_actions = n_actions
        self.values = {f"{scope}/weight": np.asarray([float(identifier)])}
        self.trainable_variable_names = tuple(self.values)
        self.learn_step_counter = 0
        self.sync_count = 0
        self.last_train = None

    def predict_eval(self, states, windows):
        base = np.arange(self.n_actions, dtype=np.float32)
        return np.tile(base, (len(states), 1))

    def predict_next(self, states, windows):
        base = np.arange(self.n_actions, dtype=np.float32) + 10.0
        return np.tile(base, (len(states), 1))

    def sync_target(self):
        self.sync_count += 1

    def train_batch(self, states, windows, targets):
        self.last_train = (
            np.array(states, copy=True),
            np.array(windows, copy=True),
            np.array(targets, copy=True),
        )
        self.values[f"{self.scope}/weight"] += 1.0
        return float(np.mean(targets**2))

    def snapshot_variables(self):
        return {name: np.array(value, copy=True) for name, value in self.values.items()}


class _FakeRuntime:
    instances = []

    def __init__(self, seed=None):
        self.seed = seed
        self.agents = []
        self.initialized = False
        self.closed = False
        self.__class__.instances.append(self)

    def build_agent(self, scope, **kwargs):
        agent = _FakeNetwork(scope, len(self.agents), kwargs["n_actions"])
        self.agents.append(agent)
        return agent

    def initialize(self):
        self.initialized = True

    def metadata(self):
        return {"agent_count": len(self.agents), "seed": self.seed}

    def close(self):
        self.closed = True


class _FailingRuntime(_FakeRuntime):
    def build_agent(self, scope, **kwargs):
        if len(self.agents) == 1:
            raise RuntimeError("synthetic build failure")
        return super().build_agent(scope, **kwargs)


def _config(**overrides):
    values = dict(
        pool_users=2,
        total_steps=5,
        batch_size=2,
        memory_size=8,
        history_steps=3,
        learning_rate=0.01,
        gamma=0.5,
        target_update=2,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _transition(
    policy="fixed",
    marker=1.0,
    action=1,
    done=False,
    context_bin=0,
    next_task=None,
):
    state = np.asarray([marker, marker + 0.1], dtype=np.float32)
    next_state = state + 1.0
    if next_task is not None:
        next_state[0] = next_task
    window = np.asarray(
        [[marker, 0.0], [marker + 1.0, 0.0], [marker + 2.0, 0.0]],
        dtype=np.float32,
    )
    return learner.Transition(
        origin_expert=policy,
        origin_energy_weight=1.2,
        state=state,
        history_window=window,
        action=action,
        next_state=next_state,
        next_history_window=window + 1.0,
        origin_done=done,
        context_bin=context_bin,
    )


class SpecialistBankTests(unittest.TestCase):
    def setUp(self):
        _FakeRuntime.instances.clear()
        patcher = mock.patch.object(learner, "_load_runtime_class", return_value=_FakeRuntime)
        self.addCleanup(patcher.stop)
        patcher.start()

    def test_builds_disjoint_network_and_replay_for_every_policy_and_ue(self):
        bank = learner.SpecialistBank(_config(), 17, 4, 2, 2)
        runtime = _FakeRuntime.instances[-1]
        self.assertTrue(runtime.initialized)
        self.assertEqual(len(runtime.agents), 6)
        self.assertEqual(len({id(agent) for agent in runtime.agents}), 6)
        self.assertEqual(
            {policy: len(bank._slots[policy]) for policy in learner.POLICIES},
            {"fixed": 2, "adaptive": 2, "contextual": 2},
        )
        replay_ids = {
            id(slot.replay) for slots in bank._slots.values() for slot in slots
        }
        self.assertEqual(len(replay_ids), 6)

    def test_zero_exploration_is_deterministic_and_mutation_free(self):
        bank = learner.SpecialistBank(_config(), 3, 4, 2, 2)
        before = bank.snapshot()
        actions = [
            bank.choose("fixed", 0, np.ones(2), np.ones((3, 2)))
            for _ in range(5)
        ]
        self.assertEqual(actions, [3] * 5)
        self.assertEqual(bank.snapshot(), before)

    def test_choose_rejects_non_finite_q_values(self):
        bank = learner.SpecialistBank(_config(), 3, 4, 2, 2)
        network = bank._slots["fixed"][0].network
        network.predict_eval = lambda states, windows: np.full(
            (len(states), 4), np.nan
        )
        with self.assertRaisesRegex(FloatingPointError, "Q values"):
            bank.choose("fixed", 0, np.ones(2), np.ones((3, 2)))

    def test_exploration_samples_local_action_too(self):
        bank = learner.SpecialistBank(_config(), 2, 4, 2, 2)
        actions = {
            bank.choose("fixed", 0, np.ones(2), np.ones((3, 2)), exploration=1)
            for _ in range(100)
        }
        self.assertEqual(actions, {0, 1, 2, 3})

    def test_store_rejects_wrong_origin_and_copies_explicit_windows(self):
        bank = learner.SpecialistBank(_config(), 1, 4, 2, 2)
        transition = _transition()
        with self.assertRaisesRegex(ValueError, "originated"):
            bank.store("adaptive", 0, transition, 2.0)
        bank.store("fixed", 0, transition, 2.0)
        transition.history_window[:] = 99.0
        stored = bank._slots["fixed"][0].replay[0].transition
        np.testing.assert_array_equal(stored.history_window[:, 0], [1.0, 2.0, 3.0])

    def test_double_dqn_target_uses_terminal_mask_and_chronological_windows(self):
        bank = learner.SpecialistBank(
            _config(pool_users=1, batch_size=3), 5, 4, 2, 2
        )
        bank.store("fixed", 0, _transition(marker=1.0, action=0, done=True), 7.0)
        bank.store("fixed", 0, _transition(marker=10.0, action=1, done=False), 2.0)
        bank.store(
            "fixed",
            0,
            _transition(marker=20.0, action=2, done=False, next_task=0.0),
            3.0,
        )
        result = bank.learn("fixed")
        self.assertEqual(result["agents_updated"], 1)
        network = bank._slots["fixed"][0].network
        states, windows, targets = network.last_train
        by_marker = {
            float(state[0]): (window, target)
            for state, window, target in zip(states, windows, targets)
        }
        terminal_window, terminal_target = by_marker[1.0]
        continuing_window, continuing_target = by_marker[10.0]
        idle_window, idle_target = by_marker[20.0]
        np.testing.assert_array_equal(terminal_window[:, 0], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(continuing_window[:, 0], [10.0, 11.0, 12.0])
        self.assertAlmostEqual(float(terminal_target[0]), 7.0)
        # eval argmax is action 3 and target Q(action 3) is 13.
        self.assertAlmostEqual(float(continuing_target[1]), 2.0 + 0.5 * 13.0)
        np.testing.assert_array_equal(idle_window[:, 0], [20.0, 21.0, 22.0])
        # No task means action 0 is the only legal bootstrap action (target Q=10).
        self.assertAlmostEqual(float(idle_target[2]), 3.0 + 0.5 * 10.0)
        self.assertEqual(network.sync_count, 1)

    def test_learning_rejects_non_finite_q_arrays(self):
        bank = learner.SpecialistBank(_config(pool_users=1), 19, 4, 2, 2)
        for marker in (1.0, 2.0):
            bank.store("fixed", 0, _transition(marker=marker), marker)
        network = bank._slots["fixed"][0].network
        network.predict_next = lambda states, windows: np.full(
            (len(states), 4), np.inf
        )
        with self.assertRaisesRegex(FloatingPointError, "q_target_next"):
            bank.learn("fixed")

    def test_learning_rejects_non_finite_loss(self):
        bank = learner.SpecialistBank(_config(pool_users=1), 29, 4, 2, 2)
        for marker in (1.0, 2.0):
            bank.store("fixed", 0, _transition(marker=marker), marker)
        bank._slots["fixed"][0].network.train_batch = (
            lambda states, windows, targets: np.nan
        )
        with self.assertRaisesRegex(FloatingPointError, "training loss"):
            bank.learn("fixed")

    def test_snapshot_rejects_non_finite_variable(self):
        bank = learner.SpecialistBank(_config(pool_users=1), 31, 4, 2, 2)
        network = bank._slots["adaptive"][0].network
        network.values[f"{network.scope}/weight"][0] = np.inf
        with self.assertRaisesRegex(FloatingPointError, "snapshot variable"):
            bank.snapshot()

    def test_learning_one_policy_preserves_other_policy_state(self):
        bank = learner.SpecialistBank(_config(pool_users=1), 11, 4, 2, 2)
        for marker in (1.0, 2.0):
            bank.store("fixed", 0, _transition(marker=marker), marker)
        before = bank.snapshot()
        bank.learn("fixed")
        after = bank.snapshot()
        self.assertNotEqual(
            before["policies"]["fixed"]["weight_digest"],
            after["policies"]["fixed"]["weight_digest"],
        )
        self.assertNotEqual(
            before["policies"]["fixed"]["trainable_weight_digest"],
            after["policies"]["fixed"]["trainable_weight_digest"],
        )
        for policy in ("adaptive", "contextual"):
            self.assertEqual(before["policies"][policy], after["policies"][policy])
        self.assertEqual(after["policies"]["fixed"]["update_count"], 1)
        self.assertEqual(after["policies"]["fixed"]["replay_count"], 2)
        self.assertEqual(after["policies"]["fixed"]["network_count"], 1)
        self.assertEqual(after["schema_version"], 1)

    def test_returning_to_policy_reuses_its_learned_network(self):
        bank = learner.SpecialistBank(_config(pool_users=1), 13, 4, 2, 2)
        for marker in (1.0, 2.0):
            bank.store("fixed", 0, _transition(marker=marker), marker)
        bank.learn("fixed")
        learned_digest = bank.snapshot()["policies"]["fixed"]["weight_digest"]
        bank.choose("adaptive", 0, np.ones(2), np.ones((3, 2)))
        bank.choose("fixed", 0, np.ones(2), np.ones((3, 2)))
        self.assertEqual(
            bank.snapshot()["policies"]["fixed"]["weight_digest"],
            learned_digest,
        )

    def test_close_is_idempotent(self):
        bank = learner.SpecialistBank(_config(), 7, 4, 2, 2)
        bank.close()
        bank.close()
        self.assertTrue(_FakeRuntime.instances[-1].closed)

    def test_partial_build_failure_closes_runtime(self):
        with mock.patch.object(
            learner, "_load_runtime_class", return_value=_FailingRuntime
        ):
            with self.assertRaisesRegex(RuntimeError, "synthetic build failure"):
                learner.SpecialistBank(_config(), 7, 4, 2, 2)
        self.assertTrue(_FailingRuntime.instances[-1].closed)


@unittest.skipUnless(
    os.environ.get("QECO_RUN_TF_TESTS") == "1",
    "set QECO_RUN_TF_TESTS=1 in the pinned TensorFlow runtime",
)
class RealTensorFlowSpecialistBankTests(unittest.TestCase):
    def test_real_optimizer_updates_only_selected_policy(self):
        # Restore the real loader because the fake patch is scoped to the other class.
        bank = learner.SpecialistBank(_config(pool_users=1), 23, 4, 2, 2)
        try:
            bank.store("fixed", 0, _transition(marker=1.0), 1.0)
            idle_transition = _transition(marker=2.0, next_task=0.0)
            bank.store("fixed", 0, idle_transition, 2.0)
            before = bank.snapshot()
            slot = bank._slots["fixed"][0]
            bank._sync_target(slot.network)
            target_q_action_zero = bank._predict_next(
                slot.network,
                idle_transition.next_state[np.newaxis, :],
                idle_transition.next_history_window[np.newaxis, :, :],
            )[0, 0]
            captured = {}
            real_optimize = bank._optimize

            def capture_and_optimize(network, states, windows, targets):
                captured["states"] = np.array(states, copy=True)
                captured["targets"] = np.array(targets, copy=True)
                return real_optimize(network, states, windows, targets)

            bank._optimize = capture_and_optimize
            result = bank.learn("fixed")
            after = bank.snapshot()
            self.assertEqual(result["agents_updated"], 1)
            self.assertNotEqual(
                before["policies"]["fixed"]["trainable_weight_digest"],
                after["policies"]["fixed"]["trainable_weight_digest"],
            )
            self.assertEqual(
                before["policies"]["adaptive"]["trainable_weight_digest"],
                after["policies"]["adaptive"]["trainable_weight_digest"],
            )
            idle_index = int(np.flatnonzero(captured["states"][:, 0] == 2.0)[0])
            self.assertAlmostEqual(
                float(captured["targets"][idle_index, 1]),
                2.0 + _config().gamma * float(target_q_action_zero),
                places=5,
            )
        finally:
            bank.close()


if __name__ == "__main__":
    unittest.main()
