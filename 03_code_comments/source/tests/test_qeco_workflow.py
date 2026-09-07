from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "qeco_workflow.py"
SPEC = importlib.util.spec_from_file_location("qeco_workflow_under_test", MODULE_PATH)
qeco_workflow = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = qeco_workflow
SPEC.loader.exec_module(qeco_workflow)


class QecoWorkflowTests(unittest.TestCase):
    def make_args(self, **overrides) -> SimpleNamespace:
        values = {
            "users": 800,
            "seed": 42,
            "allow_1200": False,
            "family": None,
            "density_mode": "fixed",
            "density_choices": None,
            "algorithms": None,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def populate_required_files(self, root: Path, args: SimpleNamespace) -> None:
        for path in qeco_workflow.required_files(args):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("ok\n", encoding="utf-8")

    def test_status_reports_large_load_and_runtime_problems_without_failing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            args = self.make_args()
            with mock.patch.object(qeco_workflow, "ROOT", root), mock.patch.object(
                qeco_workflow, "DEFAULT_PYTHON", root / "env_channel" / "bin" / "python"
            ), mock.patch.dict(os.environ, {"QECO_PYTHON": ""}, clear=False):
                self.populate_required_files(root, args)
                comparison = root / "results" / "comparisons" / "ablation_user_1500_seed_42_ep_500_edge_3_ap_6_channel_4"
                comparison.mkdir(parents=True)

                buffer = io.StringIO()
                with contextlib.redirect_stdout(buffer):
                    qeco_workflow.cmd_status(args)

            output = buffer.getvalue()
            self.assertIn("[retained large-load comparison evidence]", output)
            self.assertIn("ablation_user_1500_seed_42_ep_500_edge_3_ap_6_channel_4", output)
            self.assertIn("[runtime]", output)
            self.assertIn("status does not run experiments", output)

    def test_adapt2_status_requires_design_doc_not_unpublished_paper(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            args = self.make_args(family="adapt2")
            with mock.patch.object(qeco_workflow, "ROOT", root):
                required = {path.relative_to(root).as_posix() for path in qeco_workflow.required_files(args)}

            self.assertIn("docs/qeco_adapt2_design.md", required)
            self.assertFalse(any("paper_draft" in path for path in required))
            self.assertFalse(any("journals" in path for path in required))

    def test_non_adapt2_status_requires_only_executable_sources(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            args = self.make_args(family=None)
            with mock.patch.object(qeco_workflow, "ROOT", root):
                required = {path.relative_to(root).as_posix() for path in qeco_workflow.required_files(args)}

            self.assertIn("experiment_support.py", required)
            self.assertIn("qeco_runtime/D3QN.py", required)
            self.assertFalse(any(path.startswith("docs/") for path in required))

    def test_run_runtime_resolution_fails_when_env_is_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with mock.patch.object(qeco_workflow, "ROOT", root), mock.patch.object(
                qeco_workflow, "DEFAULT_PYTHON", root / "env_channel" / "bin" / "python"
            ), mock.patch.dict(os.environ, {"QECO_PYTHON": ""}, clear=False):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    qeco_workflow.ensure_python_runtime()

    def test_runtime_check_reports_missing_required_modules(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runtime_path = root / "env_channel" / "bin" / "python"
            runtime_path.parent.mkdir(parents=True)
            runtime_path.write_text("#!/bin/sh\n", encoding="utf-8")
            runtime_path.chmod(0o755)
            completed = [
                qeco_workflow.subprocess.CompletedProcess([], 0, stdout="3.10.13\n", stderr=""),
                qeco_workflow.subprocess.CompletedProcess([], 0, stdout="tensorflow\n", stderr=""),
            ]
            with mock.patch.object(qeco_workflow, "ROOT", root), mock.patch.object(
                qeco_workflow, "DEFAULT_PYTHON", runtime_path
            ), mock.patch.dict(os.environ, {"QECO_PYTHON": ""}, clear=False), mock.patch.object(
                qeco_workflow.subprocess, "run", side_effect=completed
            ):
                runtime = qeco_workflow.check_python_runtime()

            self.assertTrue(any("Missing required runtime modules: tensorflow" in issue for issue in runtime.issues))

    def test_commands_use_env_channel_path_instead_of_system_python(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            args = self.make_args(algorithms=("qeco",))
            with mock.patch.object(qeco_workflow, "ROOT", root), mock.patch.object(
                qeco_workflow, "DEFAULT_PYTHON", root / "env_channel" / "bin" / "python"
            ), mock.patch.dict(os.environ, {"QECO_PYTHON": ""}, clear=False):
                stdout = io.StringIO()
                stderr = io.StringIO()
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    qeco_workflow.cmd_commands(args)

            self.assertIn(str(root / "env_channel" / "bin" / "python"), stdout.getvalue())
            self.assertNotIn(os.path.realpath(sys.executable), stdout.getvalue())
            self.assertIn("Runtime warning", stderr.getvalue())

    def test_large_load_commands_still_require_explicit_allow_flag(self):
        with self.assertRaises(SystemExit):
            qeco_workflow.guard_users(self.make_args(users=1500))


if __name__ == "__main__":
    unittest.main()
