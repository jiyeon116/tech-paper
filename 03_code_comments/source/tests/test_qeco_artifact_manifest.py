from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import qeco_artifact_manifest as manifest


class QecoArtifactManifestTests(unittest.TestCase):
    def make_root(self) -> tempfile.TemporaryDirectory:
        temp = tempfile.TemporaryDirectory()
        root = Path(temp.name)
        (root / "results" / "run").mkdir(parents=True)
        (root / "logs").mkdir()
        (root / "results" / "run" / "QoE.txt").write_text("1.0\n", encoding="utf-8")
        (root / "logs" / "worker.log").write_text("[EXIT STATUS] 0\n", encoding="utf-8")
        (root / "results" / ".DS_Store").write_bytes(b"noise")
        (root / "results" / "__pycache__").mkdir()
        (root / "results" / "__pycache__" / "ignored.pyc").write_bytes(b"noise")
        return temp

    def test_write_manifest_covers_only_artifacts_and_ignores_noise(self):
        with self.make_root() as temp_name:
            root = Path(temp_name)
            output = root / "artifact_manifests" / "qeco_artifacts.sha256"

            manifest.write_manifest(root, output)

            text = output.read_text(encoding="utf-8")
            self.assertIn("results/run/QoE.txt", text)
            self.assertIn("logs/worker.log", text)
            self.assertIn("# Scope: results/**, logs/**", text)
            self.assertNotIn(".DS_Store", text)
            self.assertNotIn("__pycache__", text)
            self.assertEqual(manifest.verify_manifest(root, output), 0)

    def test_journal_files_are_outside_manifest_scope(self):
        with self.make_root() as temp_name:
            root = Path(temp_name)
            (root / "journals").mkdir()
            (root / "journals" / "paper.pdf").write_bytes(b"%PDF-1.7\n")

            entries = manifest.build_entries(root)

            self.assertNotIn("journals/paper.pdf", {entry.path for entry in entries})

    def test_verify_reports_missing_changed_and_extra_files(self):
        with self.make_root() as temp_name:
            root = Path(temp_name)
            output = root / "artifact_manifests" / "qeco_artifacts.sha256"
            manifest.write_manifest(root, output)

            (root / "results" / "run" / "QoE.txt").write_text("2.0\n", encoding="utf-8")
            (root / "logs" / "worker.log").unlink()
            (root / "results" / "run" / "Energy.txt").write_text("3.0\n", encoding="utf-8")

            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                status = manifest.verify_manifest(root, output)

            self.assertEqual(status, 1)
            report = buffer.getvalue()
            self.assertIn("[missing]", report)
            self.assertIn("logs/worker.log", report)
            self.assertIn("[changed]", report)
            self.assertIn("results/run/QoE.txt", report)
            self.assertIn("[extra]", report)
            self.assertIn("results/run/Energy.txt", report)

    def test_root_override_defaults_manifest_under_that_root(self):
        with self.make_root() as temp_name:
            root = Path(temp_name)
            default_output = root / "artifact_manifests" / "qeco_artifacts.sha256"

            with mock.patch("sys.argv", ["qeco_artifact_manifest.py", "--root", str(root), "write"]):
                with contextlib.redirect_stdout(io.StringIO()):
                    status = manifest.main()

            self.assertEqual(status, 0)
            self.assertTrue(default_output.is_file())
            self.assertIn("results/run/QoE.txt", default_output.read_text(encoding="utf-8"))

    def test_read_manifest_rejects_duplicate_and_unsafe_paths(self):
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            duplicate = root / "duplicate.sha256"
            duplicate.write_text(
                "\n".join(
                    [
                        manifest.MANIFEST_HEADER,
                        "0\t1\tresults/run/QoE.txt",
                        "1\t1\tresults/run/QoE.txt",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            unsafe = root / "unsafe.sha256"
            unsafe.write_text(
                "\n".join(
                    [
                        manifest.MANIFEST_HEADER,
                        "0\t1\t../outside.txt",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "duplicate path"):
                manifest.read_manifest(duplicate)
            with self.assertRaisesRegex(ValueError, "unsafe repo-relative path"):
                manifest.read_manifest(unsafe)


if __name__ == "__main__":
    unittest.main()
