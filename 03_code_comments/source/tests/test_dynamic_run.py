import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from dynamic_experiment.config import DynamicConfig
from dynamic_experiment.run import EVAL_ARMS, append_context, execute, run_episode, runtime_diagnostics
from dynamic_experiment.provenance import ProvenanceError, SourceIdentity
from dynamic_experiment.selector import DwellSelector, calibrate, context_bin
from dynamic_experiment.traces import make_trace
from experiment_support import qeco_adapt_energy_weight


class RecordingBank:
    def __init__(self):
        self.records = []
        self.calls = []
        self.learned = []

    def choose(self, policy, ue, state, window, exploration=0):
        self.calls.append((policy, ue, state.copy(), window.copy()))
        return 1 if policy == "adaptive" else 0

    def store(self, policy, ue, transition, reward):
        self.records.append((policy, ue, transition, reward))

    def learn(self, policy):
        self.learned.append(policy)

    def snapshot(self):
        return {}


class DynamicRunTests(unittest.TestCase):
    def setUp(self):
        self.config = DynamicConfig(pool_users=6, total_steps=35, load_levels=(1, 3, 6),
                                    dwell_min=2, dwell_max=5, batch_size=4)

    def test_config_validates_invariants_and_roundtrips(self):
        for kwargs in ({"pool_users": 0}, {"pool_users": True}, {"total_steps": 10},
                       {"load_levels": (1, 151)}, {"load_mode": "future"},
                       {"history_steps": 0}, {"gamma": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                DynamicConfig(**kwargs)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            path.write_text(json.dumps(self.config.payload()))
            self.assertEqual(DynamicConfig.load(path), self.config)
            self.assertEqual(DynamicConfig.load(path).digest(), self.config.digest())

    def test_context_and_dwell_are_predeclared(self):
        self.assertEqual(context_bin([0, 0, 0]), 0)
        self.assertEqual(context_bin([0.7, 0.4, 0.4]), 1)
        self.assertEqual(context_bin([1, 1, 1]), 2)
        selector = DwellSelector(("adaptive", "fixed", "adaptive"), 5)
        self.assertEqual(selector.select([0, 0, 0], 0), "adaptive")
        self.assertEqual(selector.select([0.7, 0.4, 0.4], 1), "adaptive")
        self.assertEqual(selector.select([0.7, 0.4, 0.4], 5), "fixed")
        with self.assertRaises(ValueError):
            selector.select([0, 0, 0], 5)

    def test_execute_verifies_source_before_importing_tensorflow_or_writing_results(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "result"
            with mock.patch("dynamic_experiment.run.source_identity", side_effect=ProvenanceError("dirty source")):
                with self.assertRaisesRegex(ProvenanceError, "dirty source"):
                    execute(self.config, 42, output)
            identity = SourceIdentity("archive", "a" * 40, "b" * 64, 1)
            with mock.patch("dynamic_experiment.run.source_identity", return_value=identity):
                with mock.patch.dict(os.environ, {"QECO_SOURCE_IDENTITY_HASH": "c" * 64}):
                    with self.assertRaisesRegex(RuntimeError, "differs from campaign"):
                        execute(self.config, 42, output)
            self.assertFalse(output.exists())

    def test_calibration_coverage_ties_and_collapse(self):
        records = [{"policy": policy, "context_bin": b, "qoe": 2.0}
                   for policy in ("fixed", "adaptive") for b in range(3)]
        mapping, evidence = calibrate(records, 1)
        self.assertEqual(mapping, ("fixed",) * 3)
        self.assertTrue(all(e["coverage"] == "sufficient" for e in evidence))
        mapping, evidence = calibrate(records, 2)
        self.assertTrue(all(e["coverage"] == "insufficient_fixed_default" for e in evidence))
        records.append({"policy": "adaptive", "context_bin": 0, "qoe": 4.0})
        self.assertEqual(calibrate(records, 1)[0][0], "adaptive")

    def test_context_reaches_idle_users_and_recurrent_rows(self):
        obs, recurrent = append_context(np.zeros((2, 10)), np.zeros((2, 6)), [0.2, 0.3, 0.4])
        np.testing.assert_equal(obs[:, -3:], recurrent[:, -3:])
        np.testing.assert_allclose(recurrent[:, -3:], [[0.2, 0.3, 0.4]] * 2)

    def test_origin_rewards_idle_terminal_and_windows_survive_switches(self):
        bank = RecordingBank()
        trace = make_trace(self.config, 42, "test", 0)
        scripted = tuple("adaptive" if (step // 2) % 2 == 0 else "fixed"
                         for step in range(self.config.total_steps))
        metrics, tasks = run_episode(self.config, bank, trace, "switching", training=True,
                                     diagnostic_policies=scripted)
        self.assertEqual(len(bank.records), self.config.total_steps * self.config.pool_users)
        self.assertEqual(sum(t.origin_done for _, _, t, _ in bank.records), self.config.pool_users)
        self.assertEqual(len(tasks), np.count_nonzero(trace.task_sizes))
        self.assertGreater(metrics["delayed_across_switch"], 0)
        self.assertGreater(metrics["switch_count"], 0)
        crossed = sum(any(scripted[t] != scripted[t - 1]
                          for t in range(r["origin_step"] + 1, r["completion_step"] + 1))
                      for r in tasks)
        endpoint_only = sum(scripted[r["origin_step"]] != scripted[r["completion_step"]] for r in tasks)
        self.assertEqual(metrics["delayed_across_switch"], crossed)
        self.assertGreater(crossed, endpoint_only)
        self.assertAlmostEqual(metrics["controller_seconds"],
                               metrics["context_selector_seconds"] + metrics["action_seconds"])
        self.assertAlmostEqual(metrics["controller_seconds_per_step"],
                               metrics["controller_seconds"] / self.config.total_steps)
        expected_weights = {qeco_adapt_energy_weight(int(n)) for n in trace.active_mask.sum(axis=1)}
        for policy, _, transition, reward in bank.records:
            self.assertEqual(policy, transition.origin_expert)
            self.assertIn(transition.origin_energy_weight, expected_weights)
            np.testing.assert_allclose(transition.history_window[1:], transition.next_history_window[:-1])
            if transition.state[0] == 0:
                self.assertEqual((transition.action, reward), (0, 0.0))

    def test_frozen_rollouts_do_not_store_or_learn(self):
        bank = RecordingBank()
        trace = make_trace(self.config, 77, "evaluation", 0)
        first, _ = run_episode(self.config, bank, trace, "adaptive")
        run_episode(self.config, bank, trace, "fixed")
        second, _ = run_episode(self.config, bank, trace, "adaptive")
        self.assertEqual(first["qoe"], second["qoe"])
        self.assertFalse(bank.records)
        self.assertFalse(bank.learned)

    def test_diagnostics_repeat_all_frozen_arms_with_actual_mapping(self):
        bank = RecordingBank()
        mapping = ("adaptive", "fixed", "adaptive")
        with mock.patch("dynamic_experiment.run.run_episode", wraps=run_episode) as spy:
            evidence = runtime_diagnostics(self.config, bank, 42, {}, mapping)
        self.assertEqual(evidence["arm_order_checked"], list(EVAL_ARMS))
        self.assertEqual(evidence["calibrated_mapping"], list(mapping))
        self.assertEqual([call.args[3] for call in spy.call_args_list[:8]],
                         list(EVAL_ARMS) + list(reversed(EVAL_ARMS)))
        self.assertTrue(all(call.kwargs["mapping"] == mapping for call in spy.call_args_list[:8]))
        self.assertFalse(bank.records)
        self.assertFalse(bank.learned)


if __name__ == "__main__":
    unittest.main()
