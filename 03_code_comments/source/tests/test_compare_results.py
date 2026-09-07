from __future__ import annotations

import csv
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import compare_results


class SharedExperimentPatch:
    def __init__(self, **overrides):
        self.overrides = overrides
        self.originals = {}
        self.result_root = None
        self.comparison_root = None

    def __enter__(self):
        self.result_root = compare_results.RESULT_ROOT
        self.comparison_root = compare_results.COMPARISON_ROOT
        for name, value in self.overrides.items():
            self.originals[name] = getattr(compare_results.SharedExperiment, name)
            setattr(compare_results.SharedExperiment, name, value)
        return self

    def __exit__(self, exc_type, exc, traceback):
        for name, value in self.originals.items():
            setattr(compare_results.SharedExperiment, name, value)
        compare_results.RESULT_ROOT = self.result_root
        compare_results.COMPARISON_ROOT = self.comparison_root


class CompareResultsRegressionTests(unittest.TestCase):
    def write_timeseries_csv(self, directory: Path) -> Path:
        path = directory / "comparison_timeseries.csv"
        fieldnames = ["step"] + [
            f"{algorithm}_{metric}"
            for algorithm in ("qeco", "qeco_adapt")
            for metric in compare_results.METRICS
        ]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for step in range(1, 4):
                row = {"step": step}
                for algorithm_index, algorithm in enumerate(("qeco", "qeco_adapt"), start=1):
                    for metric_index, metric in enumerate(compare_results.METRICS, start=1):
                        row[f"{algorithm}_{metric}"] = step * algorithm_index * metric_index
                writer.writerow(row)
        return path

    def write_metric_run(
        self,
        algorithm_dir: Path,
        run_name: str,
        *,
        users: int = 1500,
        seed: int = 42,
        family: str = "adapt2",
        density_mode: str = "random",
        density_choices: list[int] | None = None,
        completion_file: str | None = "CompletionRate.txt",
    ) -> Path:
        density_choices = [100, 300, 1500] if density_choices is None else density_choices
        run_dir = algorithm_dir / run_name
        run_dir.mkdir(parents=True)
        config = {
            "random_seed": seed,
            "num_users": users,
            "num_edges": 3,
            "max_delay": 10,
            "density": {
                "experiment_family": family or None,
                "mode": density_mode,
                "choices": density_choices,
            },
            "channel": {"count": 4},
            "access": {"num_access_points": 6, "ap_cluster_size": 3},
            "qeco": {"episodes": 3},
            "extra": {
                "experiment_family": family or None,
                "density_mode": density_mode,
                "density_choices": density_choices,
                "run_episodes": 3,
                "n_actions": 4,
                "n_features": 10,
                "n_lstm_state": 6,
                "action_space": "local_or_edge",
                "state_layout": "channel_aware",
            },
        }
        (run_dir / "experiment_config.json").write_text(json.dumps(config))
        for filename in ("QoE.txt", "Delay.txt", "Energy.txt", "Drop.txt"):
            (run_dir / filename).write_text("1\n2\n3\n")
        if completion_file is not None:
            values = "0.1\n0.2\n0.3\n" if completion_file == "DropRate.txt" else "0.9\n0.8\n0.7\n"
            (run_dir / completion_file).write_text(values)
        return run_dir

    def test_completed_run_requires_completion_metric(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = self.write_metric_run(Path(temp_dir), "run", completion_file=None)
            self.assertFalse(compare_results.has_metric_files(run_dir))

    def test_load_metrics_accepts_completion_rate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = self.write_metric_run(Path(temp_dir), "run")
            self.assertTrue(compare_results.has_metric_files(run_dir))
            self.assertEqual(compare_results.load_metrics(run_dir)["completion_rate"], [0.9, 0.8, 0.7])

    def test_load_metrics_derives_legacy_drop_rate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = self.write_metric_run(Path(temp_dir), "run", completion_file="DropRate.txt")
            self.assertTrue(compare_results.has_metric_files(run_dir))
            completion = compare_results.load_metrics(run_dir)["completion_rate"]
            for actual, expected in zip(completion, (0.9, 0.8, 0.7)):
                self.assertAlmostEqual(actual, expected)

    def test_load_metrics_rejects_missing_completion_metric(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = self.write_metric_run(Path(temp_dir), "run", completion_file=None)
            with self.assertRaisesRegex(
                FileNotFoundError,
                "CompletionRate.txt or legacy DropRate.txt",
            ):
                compare_results.load_metrics(run_dir)

    def test_csv_only_regeneration_preserves_values_and_section_labels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            csv_path = self.write_timeseries_csv(directory)
            existing_png = directory / "QoE_Timeseries.png"
            existing_png.write_bytes(b"existing visualization")
            legacy_png = directory / "comparison_timeseries.png"
            legacy_png.write_bytes(b"preserved legacy visualization")
            original, algorithms = compare_results.load_timeseries_csv(csv_path)
            args = SimpleNamespace(
                timeseries_csv=str(csv_path),
                output_dir=None,
                smoothing_window=25,
                no_visualize=True,
            )

            compare_results.regenerate_from_timeseries_csv(args)

            regenerated, regenerated_algorithms = compare_results.load_timeseries_csv(csv_path)
            self.assertEqual(regenerated_algorithms, algorithms)
            self.assertEqual(regenerated, original)
            summary = json.loads((directory / "comparison_summary.json").read_text())
            self.assertEqual(summary["regenerated_from_timeseries_csv"], str(csv_path.resolve()))
            self.assertIsNone(summary["regenerated_from_timeseries_csv_relative"])
            self.assertEqual(
                {Path(path) for path in summary["visualizations"]},
                {existing_png.resolve(), legacy_png.resolve()},
            )
            self.assertTrue(legacy_png.is_file())
            with (directory / "comparison_section_means.csv").open() as section_file:
                section_rows = list(csv.DictReader(section_file))
            self.assertEqual([row["section"] for row in section_rows], ["1-300", "1-300", "301-500", "301-500"])

    def test_timeseries_annotations_are_removed_but_bar_labels_remain(self):
        self.assertNotIn("annotate(", inspect.getsource(compare_results.render_timeseries))
        self.assertNotIn("annotate(", inspect.getsource(compare_results.render_selected_four_timeseries))
        self.assertIn("annotate(", inspect.getsource(compare_results.render_overall_charts))

    def test_renderers_include_non_overlapping_reading_guides(self):
        for renderer in (
            compare_results.render_timeseries,
            compare_results.render_branch_timeseries,
            compare_results.render_selected_four_timeseries,
            compare_results.render_overall_charts,
        ):
            self.assertIn("add_reading_guide(", inspect.getsource(renderer))

    def test_completion_rate_axis_uses_integer_percent_labels(self):
        self.assertEqual(
            [compare_results.format_axis_tick("completion_rate", value) for value in range(44, 49)],
            ["44%", "45%", "46%", "47%", "48%"],
        )

    def test_latest_completed_random_density_run_requires_matching_choices(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result_root = Path(temp_dir)
            algorithm_dir = result_root / "qeco"
            correct = self.write_metric_run(algorithm_dir, "run_20260824_120000", density_choices=[100, 300, 1500])
            self.write_metric_run(algorithm_dir, "run_20260825_120000", density_choices=[100, 500, 1500])

            with SharedExperimentPatch(
                NUM_USERS=1500,
                RANDOM_SEED=42,
                EXPERIMENT_FAMILY="adapt2",
                DENSITY_MODE="random",
                DENSITY_CHOICES=(100, 300, 1500),
                DENSITY_MIN=0,
                DENSITY_MAX=0,
            ):
                compare_results.RESULT_ROOT = result_root
                self.assertEqual(compare_results.latest_completed_run("qeco"), correct)

    def test_latest_completed_random_density_run_refuses_wrong_profile_fallback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result_root = Path(temp_dir)
            algorithm_dir = result_root / "qeco"
            self.write_metric_run(algorithm_dir, "run_20260825_120000", density_choices=[100, 500, 1500])

            with SharedExperimentPatch(
                NUM_USERS=1500,
                RANDOM_SEED=42,
                EXPERIMENT_FAMILY="adapt2",
                DENSITY_MODE="random",
                DENSITY_CHOICES=(100, 300, 1500),
                DENSITY_MIN=0,
                DENSITY_MAX=0,
            ):
                compare_results.RESULT_ROOT = result_root
                with self.assertRaisesRegex(FileNotFoundError, "matching current scenario"):
                    compare_results.latest_completed_run("qeco")

    def test_density_choices_match_scheduler_sorted_unique_resolution(self):
        self.assertEqual(compare_results.normalize_density_choices("300,100,300"), [100, 300])

    def test_explicit_runs_from_different_experiment_families_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            legacy = self.write_metric_run(
                root / "qeco",
                "run_legacy",
                users=800,
                family="",
                density_mode="fixed",
                density_choices=[],
            )
            adapt2 = self.write_metric_run(
                root / "qeco_adapt",
                "run_adapt2",
                users=800,
                family="adapt2",
                density_mode="fixed",
                density_choices=[],
            )

            with self.assertRaisesRegex(ValueError, "experiment_family"):
                compare_results.ensure_matching_profiles({"qeco": legacy, "qeco_adapt": adapt2})

    def test_fixed_density_run_profile_matching_keeps_legacy_compatibility(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = self.write_metric_run(
                Path(temp_dir) / "qeco",
                "run_20260824_120000",
                users=800,
                family="",
                density_mode="fixed",
                density_choices=[],
            )

            with SharedExperimentPatch(
                NUM_USERS=800,
                RANDOM_SEED=42,
                EXPERIMENT_FAMILY="",
                DENSITY_MODE="fixed",
                DENSITY_CHOICES=(),
                DENSITY_MIN=0,
                DENSITY_MAX=0,
            ):
                self.assertTrue(compare_results.run_profile_matches_current(run_dir))

    def test_comparison_summary_keeps_absolute_paths_and_adds_repo_relative_paths(self):
        with tempfile.TemporaryDirectory(dir=compare_results.ROOT_DIR) as temp_dir:
            base = Path(temp_dir)
            result_root = base / "results" / "adapt2" / "random_density"
            comparison_root = base / "results" / "adapt2" / "comparisons" / "random_density"
            for algorithm in ("qeco", "qeco_adapt"):
                self.write_metric_run(result_root / algorithm, "run_20260824_120000")

            with SharedExperimentPatch(
                NUM_USERS=1500,
                RANDOM_SEED=42,
                EXPERIMENT_FAMILY="adapt2",
                DENSITY_MODE="random",
                DENSITY_CHOICES=(100, 300, 1500),
                DENSITY_MIN=0,
                DENSITY_MAX=0,
            ):
                compare_results.RESULT_ROOT = result_root
                compare_results.COMPARISON_ROOT = comparison_root
                output_dir = compare_results.run_comparison(
                    SimpleNamespace(
                        algorithms=["qeco", "qeco_adapt"],
                        tang_wong_run=None,
                        qeco_run=None,
                        qeco_adapt_run=None,
                        output_dir=None,
                        smoothing_window=25,
                        no_visualize=True,
                    )
                )

            summary = json.loads((output_dir / "comparison_summary.json").read_text())
            self.assertTrue(summary["runs"]["qeco"].startswith(str(compare_results.ROOT_DIR)))
            self.assertEqual(
                summary["runs_relative"]["qeco"],
                (result_root / "qeco" / "run_20260824_120000").relative_to(compare_results.ROOT_DIR).as_posix(),
            )
            self.assertEqual(summary["output_dir_relative"], output_dir.relative_to(compare_results.ROOT_DIR).as_posix())
            self.assertEqual(
                summary["timeseries_csv_relative"],
                (output_dir / "comparison_timeseries.csv").relative_to(compare_results.ROOT_DIR).as_posix(),
            )


if __name__ == "__main__":
    unittest.main()
