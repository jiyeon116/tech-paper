from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from dynamic_experiment.environment import TraceMEC
from dynamic_experiment.traces import EpisodeTrace, make_trace


def _config(**overrides):
    values = dict(
        pool_users=8,
        total_steps=12,
        max_delay=3,
        load_levels=(2, 5, 8),
        load_mode="persistent",
        dwell_min=2,
        dwell_max=4,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _rng_state_equal(first, second):
    return (
        first[0] == second[0]
        and np.array_equal(first[1], second[1])
        and first[2:] == second[2:]
    )


class EpisodeTraceTests(unittest.TestCase):
    def test_reproducible_phase_disjoint_trace_and_digest(self):
        config = _config()
        first = make_trace(config, seed=42, phase="train", episode=3)
        second = make_trace(config, seed=42, phase="train", episode=3)
        calibration = make_trace(config, seed=42, phase="calibration", episode=3)
        self.assertEqual(first.digest(), second.digest())
        self.assertNotEqual(first.digest(), calibration.digest())
        self.assertEqual(len(first.digest()), 64)
        self.assertFalse(first.active_mask.flags.writeable)
        self.assertTrue((first.task_sizes[-config.max_delay :] == 0).all())
        self.assertTrue((first.task_densities[-config.max_delay :] == 0).all())
        self.assertTrue((first.task_sizes[~first.active_mask] == 0).all())
        self.assertEqual(first.channel_gains.shape, (config.total_steps + 1, 8, 6, 4))

    def test_persistent_schedule_respects_counts_and_retains_members(self):
        config = _config(total_steps=40, dwell_min=3, dwell_max=3)
        trace = make_trace(config, seed=7, phase="evaluation", episode=0)
        counts = trace.active_mask.sum(axis=1)
        self.assertTrue(set(counts) <= set(config.load_levels))
        for step in range(1, config.total_steps):
            previous = trace.active_mask[step - 1]
            current = trace.active_mask[step]
            if counts[step] == counts[step - 1]:
                np.testing.assert_array_equal(current, previous)
            else:
                self.assertEqual(
                    int(np.count_nonzero(previous & current)),
                    int(min(counts[step - 1], counts[step])),
                )

    def test_iid_schedule_changes_independently_and_rejects_invalid_config(self):
        trace = make_trace(_config(load_mode="iid", total_steps=40), 5, "train", 1)
        self.assertGreater(len(set(trace.active_mask.sum(axis=1))), 1)
        with self.assertRaisesRegex(ValueError, "load_levels"):
            make_trace(_config(load_levels=(9,)), 5, "train", 1)
        with self.assertRaisesRegex(ValueError, "total_steps"):
            make_trace(_config(total_steps=3, max_delay=3), 5, "train", 1)

    def test_trace_generation_does_not_consume_global_numpy_rng(self):
        np.random.seed(1234)
        before = np.random.get_state()
        make_trace(_config(), seed=99, phase="train", episode=2)
        after = np.random.get_state()
        self.assertTrue(_rng_state_equal(before, after))


class TraceMECTests(unittest.TestCase):
    def _manual_trace(self):
        steps = 5
        users = 2
        mask = np.asarray(
            [[True, False], [False, True], [False, False], [False, False], [False, False]]
        )
        sizes = np.zeros((steps, users))
        densities = np.zeros((steps, users))
        sizes[0, 0] = 6.0
        densities[0, 0] = 0.397
        gains = np.full((steps + 1, users, 6, 4), 5e-6)
        gains[:, 1, :, :] = 1e-7
        return EpisodeTrace(
            active_mask=mask,
            task_sizes=sizes,
            task_densities=densities,
            ue_energy=np.asarray([0.25, 0.75]),
            fronthaul=np.linspace(0.01, 0.06, 6),
            channel_gains=gains,
        )

    def test_context_uses_current_mask_backlog_and_all_users_raw_channels(self):
        trace = self._manual_trace()
        env = TraceMEC(trace, max_delay=3)
        env.reset_trace()
        env.b_edge_comp[:] = 0
        env.b_edge_comp[0, 0] = 12
        context = env.context()
        self.assertAlmostEqual(context[0], 0.5)
        self.assertAlmostEqual(context[1], 1.0 / 3.0)
        best = np.max(env._channel_rate(trace.channel_gains[0]), axis=(1, 2))
        expected_pressure = 1.0 - np.mean([env._normalized_rate(value) for value in best])
        self.assertAlmostEqual(context[2], expected_pressure)

    def test_deactivation_does_not_purge_inflight_local_work(self):
        trace = self._manual_trace()
        env = TraceMEC(trace, max_delay=3)
        env.reset_trace()
        env.step(np.zeros(trace.pool_users, dtype=int))
        remaining_after_arrival = env.local_process_task[0]["REMAIN"]
        self.assertGreater(remaining_after_arrival, 0)
        env.step(np.zeros(trace.pool_users, dtype=int))
        self.assertLess(env.local_process_task[0]["REMAIN"], remaining_after_arrival)

    def test_deactivation_does_not_purge_offloaded_work(self):
        trace = self._manual_trace()
        env = TraceMEC(trace, max_delay=3)
        env.reset_trace()
        env.comp_cap_edge[:] = 0.1
        env.step(np.asarray([1, 0], dtype=int))
        self.assertFalse(env.edge_computation_queue[0][0].empty())
        env.step(np.zeros(trace.pool_users, dtype=int))
        self.assertGreater(env.edge_process_task[0][0]["REMAIN"], 0)
        self.assertTrue(env.trace.active_mask[1, 0] == 0)

    def test_backlog_observation_is_recomputed_from_queue_and_in_service(self):
        trace = self._manual_trace()
        env = TraceMEC(trace, max_delay=3)
        env.reset_trace()
        env.edge_computation_queue[0][1].put({"SIZE": 4.0})
        env.edge_process_task[1][2]["REMAIN"] = 3.0
        base_obs = np.zeros((trace.pool_users, env.base_n_features))
        base_obs[0, 0] = 1.0
        env._repair_edge_backlog(base_obs)
        self.assertEqual(env.b_edge_comp[0, 1], 4.0)
        self.assertEqual(env.b_edge_comp[1, 2], 3.0)
        self.assertEqual(base_obs[0, 4], 4.0)

    def test_backlog_rejects_malformed_queued_task(self):
        env = TraceMEC(self._manual_trace(), max_delay=3)
        env.reset_trace()
        env.edge_computation_queue[0][0].put({"TASK_ID": 7})
        with self.assertRaisesRegex(ValueError, "REMAIN or SIZE"):
            env._outstanding_edge_bits()

    def test_environment_operations_preserve_global_numpy_rng_and_classify_arrivals(self):
        trace = self._manual_trace()
        np.random.seed(765)
        before = np.random.get_state()
        env = TraceMEC(trace, max_delay=3)
        obs, recurrent = env.reset_trace()
        self.assertTrue(_rng_state_equal(before, np.random.get_state()))
        self.assertEqual(obs.shape, (2, env.n_features))
        self.assertEqual(recurrent.shape, (2, env.n_lstm_state))
        done = False
        while not done:
            _, _, done = env.step(np.zeros(trace.pool_users, dtype=int))
        self.assertTrue(_rng_state_equal(before, np.random.get_state()))
        self.assertTrue(np.array_equal(env.arrive_task_size != 0, env.process_delay > 0))

    def test_empty_population_episode_has_finite_zero_metrics(self):
        from experiment_support import compute_env_metrics

        config = _config(pool_users=3, load_levels=(0,))
        trace = make_trace(config, seed=1, phase="evaluation", episode=0)
        env = TraceMEC(trace, max_delay=config.max_delay)
        env.reset_trace()
        done = False
        while not done:
            _, _, done = env.step(np.zeros(config.pool_users, dtype=int))
        metrics = compute_env_metrics(env, np)
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
        self.assertEqual(metrics["arrived_task"], 0.0)
        self.assertEqual(metrics["qoe"], 0.0)


if __name__ == "__main__":
    unittest.main()
