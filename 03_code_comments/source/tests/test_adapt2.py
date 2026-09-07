"""Unit tests for the qeco-adapt2 extension.

These tests exercise pure helpers (density regimes, density schedule, result
folder routing, adapter mode selection) and never build TensorFlow agents.
Run from the repository root:

    python -m unittest tests/test_adapt2.py
"""
from __future__ import annotations

import tempfile
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import experiment_support as es
from algorithm_adapters import registry


class _SharedExperimentPatch:
    """Context manager that temporarily overrides SharedExperiment class attributes."""

    def __init__(self, **overrides):
        self.overrides = overrides
        self.saved = {}

    def __enter__(self):
        for key, value in self.overrides.items():
            self.saved[key] = getattr(es.SharedExperiment, key)
            setattr(es.SharedExperiment, key, value)
        return self

    def __exit__(self, *exc):
        for key, value in self.saved.items():
            setattr(es.SharedExperiment, key, value)
        return False


class GlobalSeedTests(unittest.TestCase):
    def test_applies_python_and_numpy_seed(self):
        with mock.patch.object(es.random, "seed") as python_seed, mock.patch.object(
            np.random, "seed"
        ) as numpy_seed:
            es.apply_global_seed(17)
        python_seed.assert_called_once_with(17)
        numpy_seed.assert_called_once_with(17)

    def test_numpy_seed_failure_is_reported(self):
        with mock.patch.object(np.random, "seed", side_effect=ValueError("broken")):
            with self.assertRaisesRegex(RuntimeError, "NumPy random seed 17"):
                es.apply_global_seed(17)

    def test_missing_numpy_dependency_is_reported(self):
        real_import = __import__

        def rejecting_import(name, *args, **kwargs):
            if name == "numpy":
                raise ModuleNotFoundError("numpy unavailable")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=rejecting_import):
            with self.assertRaisesRegex(RuntimeError, "NumPy random seed 17"):
                es.apply_global_seed(17)

    def test_requested_tensorflow_import_failure_is_reported(self):
        real_import = __import__

        def rejecting_import(name, *args, **kwargs):
            if name == "tensorflow.compat.v1":
                raise ModuleNotFoundError("tensorflow unavailable")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=rejecting_import):
            with self.assertRaisesRegex(RuntimeError, "TensorFlow random seed 17"):
                es.apply_global_seed(17, use_tensorflow=True)


class DensityRegimeTests(unittest.TestCase):
    def test_default_thresholds_map_tested_user_counts(self):
        with _SharedExperimentPatch(
            NUM_EDGES=3,
            ADAPT2_SPARSE_MAX_USERS_PER_EDGE=50.0,
            ADAPT2_DENSE_MIN_USERS_PER_EDGE=350.0,
        ):
            self.assertEqual(es.classify_density_regime(30), "sparse")
            self.assertEqual(es.classify_density_regime(100), "sparse")
            self.assertEqual(es.classify_density_regime(149), "sparse")
            self.assertEqual(es.classify_density_regime(150), "general")  # 50 users/edge is not < 50
            self.assertEqual(es.classify_density_regime(300), "general")
            self.assertEqual(es.classify_density_regime(800), "general")
            self.assertEqual(es.classify_density_regime(1049), "general")
            self.assertEqual(es.classify_density_regime(1050), "dense")  # 350 users/edge
            self.assertEqual(es.classify_density_regime(1200), "dense")
            self.assertEqual(es.classify_density_regime(1500), "dense")

    def test_regime_policy_map_uses_configured_names(self):
        with _SharedExperimentPatch(
            ADAPT2_SPARSE_POLICY="qeco",
            ADAPT2_GENERAL_POLICY="qeco-adapt-general",
            ADAPT2_DENSE_POLICY="qeco-adaptive-gate",
        ):
            self.assertEqual(
                es.adapt2_regime_policy_map(),
                {"sparse": "qeco", "general": "qeco-adapt-general", "dense": "qeco-adaptive-gate"},
            )

    def test_load_terms_follow_active_users_override(self):
        with _SharedExperimentPatch(NUM_USERS=800, NUM_EDGES=3):
            default_strength = es.qeco_adapt_gating_strength()
            self.assertAlmostEqual(default_strength, es.qeco_adapt_gating_strength(800))
            self.assertLess(es.qeco_adapt_gating_strength(100), default_strength)
            self.assertGreater(es.qeco_adapt_gating_strength(1500), default_strength)
            self.assertLess(es.qeco_adapt_energy_weight(100), es.qeco_adapt_energy_weight(1500))
            # Values quoted in the extended draft, Section 7.2 table.
            self.assertAlmostEqual(es.qeco_adapt_gating_strength(800), 0.7170, places=4)
            self.assertAlmostEqual(es.qeco_adapt_energy_weight(800), 1.4499, places=4)
            self.assertAlmostEqual(es.qeco_adapt_gating_strength(1500), 0.8261, places=4)


class DensityScheduleTests(unittest.TestCase):
    def test_fixed_mode_uses_pool_every_episode(self):
        with _SharedExperimentPatch(DENSITY_MODE="fixed", DENSITY_SUBSET="random"):
            schedule = es.DensitySchedule(5, agent_pool_users=40, seed=42)
            self.assertEqual(schedule.active_users, [40] * 5)
            self.assertEqual(schedule.choices, (40,))
            self.assertTrue(schedule.active_mask(0).all())

    def test_random_mode_is_reproducible_and_bounded(self):
        with _SharedExperimentPatch(
            DENSITY_MODE="random",
            DENSITY_CHOICES=(10, 20, 30),
            DENSITY_MIN=0,
            DENSITY_MAX=0,
            DENSITY_SUBSET="random",
        ):
            first = es.DensitySchedule(50, agent_pool_users=30, seed=7)
            second = es.DensitySchedule(50, agent_pool_users=30, seed=7)
            other = es.DensitySchedule(50, agent_pool_users=30, seed=8)
            self.assertEqual(first.active_users, second.active_users)
            self.assertNotEqual(first.active_users, other.active_users)
            self.assertTrue(set(first.active_users) <= {10, 20, 30})
            for episode, count in enumerate(first.active_users):
                mask = first.active_mask(episode)
                self.assertEqual(int(mask.sum()), count)
                np.testing.assert_array_equal(mask, second.active_mask(episode))
            self.assertEqual(len(first.regimes()), 50)

    def test_random_mode_rejects_choices_above_pool(self):
        with _SharedExperimentPatch(
            DENSITY_MODE="random",
            DENSITY_CHOICES=(100, 300),
            DENSITY_MIN=0,
            DENSITY_MAX=0,
        ):
            with self.assertRaises(ValueError):
                es.DensitySchedule(3, agent_pool_users=200, seed=1)

    def test_random_mode_requires_choices_or_range(self):
        with _SharedExperimentPatch(DENSITY_MODE="random", DENSITY_CHOICES=(), DENSITY_MIN=0, DENSITY_MAX=0):
            with self.assertRaises(ValueError):
                es.DensitySchedule(3, agent_pool_users=100, seed=1)
        with _SharedExperimentPatch(DENSITY_MODE="random", DENSITY_CHOICES=(), DENSITY_MIN=5, DENSITY_MAX=8):
            schedule = es.DensitySchedule(10, agent_pool_users=8, seed=1)
            self.assertEqual(schedule.choices, (5, 6, 7, 8))

    def test_prefix_subset_activates_lowest_indices(self):
        with _SharedExperimentPatch(DENSITY_MODE="random", DENSITY_CHOICES=(2, 4), DENSITY_SUBSET="prefix"):
            schedule = es.DensitySchedule(6, agent_pool_users=4, seed=3)
            for episode, count in enumerate(schedule.active_users):
                mask = schedule.active_mask(episode)
                self.assertTrue(mask[:count].all())
                self.assertFalse(mask[count:].any())

    def test_active_user_mask_zeroes_inactive_arrivals(self):
        size = np.ones((5, 4))
        dens = np.ones((5, 4))
        mask = np.array([True, False, True, False])
        size, dens = es.apply_active_user_mask(size, dens, mask)
        self.assertTrue((size[:, [0, 2]] == 1).all())
        self.assertTrue((size[:, [1, 3]] == 0).all())
        self.assertTrue((dens[:, [1, 3]] == 0).all())


class ResultFolderRoutingTests(unittest.TestCase):
    def test_legacy_layout_when_no_family(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with _SharedExperimentPatch(RESULT_ROOT=root, EXPERIMENT_FAMILY="", DENSITY_MODE="fixed"):
                run_dir = es.create_run_dir("qeco_adapt2")
                self.assertEqual(run_dir.parent, root / "qeco_adapt2")
                self.assertEqual(es.SharedExperiment.comparison_base_dir(), root / "comparisons")

    def test_same_second_run_dirs_do_not_collide(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with _SharedExperimentPatch(RESULT_ROOT=root, EXPERIMENT_FAMILY="adapt2", DENSITY_MODE="fixed"):
                original_strftime = es.time.strftime
                es.time.strftime = lambda fmt: "run_20260820_120000"
                try:
                    first = es.create_run_dir("qeco_adapt_general")
                    second = es.create_run_dir("qeco_adapt_general")
                finally:
                    es.time.strftime = original_strftime
                self.assertNotEqual(first, second)
                self.assertEqual(first.name, "run_20260820_120000")
                self.assertEqual(second.name, "run_20260820_120000_2")
                self.assertTrue(first.exists() and second.exists())

    def test_family_layout_is_classified_by_density_mode(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with _SharedExperimentPatch(RESULT_ROOT=root, EXPERIMENT_FAMILY="adapt2", DENSITY_MODE="random"):
                run_dir = es.create_run_dir("qeco_adapt2")
                self.assertEqual(run_dir.parent, root / "adapt2" / "random_density" / "qeco_adapt2")
                self.assertEqual(
                    es.SharedExperiment.comparison_base_dir(),
                    root / "adapt2" / "comparisons" / "random_density",
                )
            with _SharedExperimentPatch(RESULT_ROOT=root, EXPERIMENT_FAMILY="adapt2", DENSITY_MODE="fixed"):
                run_dir = es.create_run_dir("qeco_adapt_general")
                self.assertEqual(run_dir.parent, root / "adapt2" / "fixed_density" / "qeco_adapt_general")


class Adapt2AdapterTests(unittest.TestCase):
    env = SimpleNamespace(n_edge=3, n_actions=4, n_features=10, n_lstm_state=6, n_time=110, base_n_features=7)
    config = SimpleNamespace(N_UE=800)

    def test_aliases_share_modes_with_their_originals(self):
        general = registry.QecoAdaptGeneralAdapter(self.env, self.config)
        original = registry.QecoAdaptiveWeightFixedGateAdapter(self.env, self.config)
        self.assertEqual((general.energy_weight_mode, general.gate_mode), (original.energy_weight_mode, original.gate_mode))
        self.assertEqual((general.energy_weight_mode, general.gate_mode), ("adaptive", "fixed"))

        adaptive_gate = registry.QecoAdaptiveGateAdapter(self.env, self.config)
        full = registry.QecoAdaptAdapter(self.env, self.config)
        self.assertEqual((adaptive_gate.energy_weight_mode, adaptive_gate.gate_mode), (full.energy_weight_mode, full.gate_mode))
        self.assertEqual((adaptive_gate.energy_weight_mode, adaptive_gate.gate_mode), ("adaptive", "adaptive"))

    def test_registry_exposes_adapt2_family(self):
        names = registry.available_channel_algorithm_names()
        for name in ("qeco-adapt-general", "qeco-adaptive-gate", "qeco-adapt2"):
            self.assertIn(name, names)
        self.assertEqual(registry.ADAPT2_FAMILY_ALGORITHMS, ("qeco", "qeco-adapt-general", "qeco-adaptive-gate", "qeco-adapt2"))

    def test_adapt2_switches_modes_by_active_users(self):
        with _SharedExperimentPatch(
            NUM_EDGES=3,
            ADAPT2_SPARSE_MAX_USERS_PER_EDGE=50.0,
            ADAPT2_DENSE_MIN_USERS_PER_EDGE=350.0,
            ADAPT2_SPARSE_POLICY="qeco-adaptive-gate",
            ADAPT2_GENERAL_POLICY="qeco-adapt-general",
            ADAPT2_DENSE_POLICY="qeco-adaptive-gate",
        ):
            adapter = registry.QecoAdapt2Adapter(self.env, self.config)
            self.assertEqual(adapter.current_regime, "general")
            self.assertEqual((adapter.energy_weight_mode, adapter.gate_mode), ("adaptive", "fixed"))
            self.assertEqual(adapter._gate_strength(), 1.0)

            adapter.on_episode_start(1, 100)
            self.assertEqual(adapter.current_regime, "sparse")
            self.assertEqual(adapter.current_policy, "qeco-adaptive-gate")
            self.assertEqual(adapter.gate_mode, "adaptive")
            self.assertAlmostEqual(adapter._gate_strength(), es.qeco_adapt_gating_strength(100))

            adapter.on_episode_start(2, 1500)
            self.assertEqual(adapter.current_regime, "dense")
            self.assertEqual(adapter.gate_mode, "adaptive")
            self.assertAlmostEqual(adapter._gate_strength(), es.qeco_adapt_gating_strength(1500))
            self.assertAlmostEqual(adapter._energy_weight(), es.qeco_adapt_energy_weight(1500))

            selection = adapter.episode_selection()
            self.assertEqual(selection["episode"], 2)
            self.assertEqual(selection["active_users"], 1500)
            self.assertEqual(selection["regime"], "dense")
            metadata = adapter.metadata()
            self.assertIn("qeco_adapt2", metadata)
            self.assertEqual(metadata["qeco_adapt2"]["regime_policy"]["general"], "qeco-adapt-general")

    def test_adapt2_accepts_plain_qeco_and_rejects_foreign_backbones(self):
        with _SharedExperimentPatch(ADAPT2_SPARSE_POLICY="qeco"):
            adapter = registry.QecoAdapt2Adapter(self.env, self.config)
            adapter.on_episode_start(0, 30)
            self.assertEqual((adapter.energy_weight_mode, adapter.gate_mode), ("qeco", "none"))
        with _SharedExperimentPatch(ADAPT2_SPARSE_POLICY="tang-wong"):
            with self.assertRaises(ValueError):
                registry.QecoAdapt2Adapter(self.env, self.config)
        with _SharedExperimentPatch(ADAPT2_SPARSE_POLICY="qeco-adapt2"):
            with self.assertRaises(ValueError):
                registry.QecoAdapt2Adapter(self.env, self.config)

    def test_non_switching_adapters_still_track_active_users(self):
        adapter = registry.QecoAdaptiveGateAdapter(self.env, self.config)
        adapter.on_episode_start(3, 100)
        self.assertEqual(adapter.active_users, 100)
        self.assertAlmostEqual(adapter._gate_strength(), es.qeco_adapt_gating_strength(100))
        self.assertIsNone(adapter.episode_selection())


class CompareResultsAdapt2Tests(unittest.TestCase):
    def test_scenario_name_and_regime_helper(self):
        import compare_results

        fixed_profile = {
            "num_users": 300,
            "random_seed": 42,
            "episodes": 500,
            "num_edges": 3,
            "access_points": 6,
            "channels": 4,
            "density_mode": "fixed",
            "density_choices": [],
        }
        self.assertEqual(compare_results.scenario_name(fixed_profile), "user_300_seed_42_ep_500_edge_3_ap_6_channel_4")
        random_profile = dict(fixed_profile, num_users=1500, density_mode="random", density_choices=[100, 300, 1500])
        self.assertEqual(
            compare_results.scenario_name(random_profile),
            "user_100-1500_random_seed_42_ep_500_edge_3_ap_6_channel_4",
        )
        context = {"num_edges": 3, "sparse_max_users_per_edge": 50.0, "dense_min_users_per_edge": 350.0}
        self.assertEqual(compare_results.regime_for_active_users(100, context), "sparse")
        self.assertEqual(compare_results.regime_for_active_users(800, context), "general")
        self.assertEqual(compare_results.regime_for_active_users(1500, context), "dense")
        self.assertIn(compare_results.SELECTED_FOUR_ADAPT2_ALGORITHMS, compare_results.SELECTED_FOUR_SETS)
        self.assertEqual(
            compare_results.selected_four_algorithms(("qeco", "qeco_adapt_general", "qeco_adaptive_gate", "qeco_adapt2")),
            compare_results.SELECTED_FOUR_ADAPT2_ALGORITHMS,
        )


if __name__ == "__main__":
    unittest.main()
