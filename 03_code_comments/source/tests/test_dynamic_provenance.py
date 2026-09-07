from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from dynamic_experiment.provenance import (
    MANIFEST_NAME,
    REVISION_NAME,
    ProvenanceError,
    create_archive_manifest,
    source_identity,
)


REVISION = "1" * 40


class DynamicProvenanceTests(unittest.TestCase):
    def test_archive_identity_detects_changed_missing_and_extra_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "dynamic_experiment" / "module.py"
            source.parent.mkdir()
            source.write_text("VALUE = 1\n", encoding="utf-8")
            created = create_archive_manifest(root, REVISION)

            verified = source_identity(root)
            self.assertEqual(verified, created)
            self.assertEqual(verified.mode, "archive")

            source.write_text("VALUE = 2\n", encoding="utf-8")
            with self.assertRaisesRegex(ProvenanceError, "hashes differ"):
                source_identity(root)
            source.write_text("VALUE = 1\n", encoding="utf-8")

            extra = root / "unexpected.py"
            extra.write_text("pass\n", encoding="utf-8")
            with self.assertRaisesRegex(ProvenanceError, "file set differs"):
                source_identity(root)

    def test_archive_ignores_runtime_results_and_rejects_revision_rewrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "main.py").write_text("pass\n", encoding="utf-8")
            identity = create_archive_manifest(root, REVISION)
            result = root / "results" / "seed_42"
            result.mkdir(parents=True)
            (result / "summary.json").write_text("{}\n", encoding="utf-8")
            (root / ".omx").mkdir()
            (root / ".omx" / "state.json").write_text("{}\n", encoding="utf-8")
            self.assertEqual(source_identity(root), identity)

            with self.assertRaisesRegex(ProvenanceError, "existing REVISION differs"):
                create_archive_manifest(root, "2" * 40)

    def test_archive_rejects_unsafe_manifest_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "main.py").write_text("pass\n", encoding="utf-8")
            create_archive_manifest(root, REVISION)
            manifest_path = root / MANIFEST_NAME
            manifest = json.loads(manifest_path.read_text())
            manifest["files"]["../outside.py"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ProvenanceError):
                source_identity(root)

    def test_clean_git_identity_rejects_tracked_and_untracked_source_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            subprocess.run(
                ["git", "-C", str(root), "config", "user.email", "test@example.invalid"],
                check=True,
            )
            source = root / "main.py"
            source.write_text("VALUE = 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "main.py"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)

            clean = source_identity(root)
            self.assertEqual(clean.mode, "git")
            self.assertEqual(clean.file_count, 1)

            source.write_text("VALUE = 2\n", encoding="utf-8")
            with self.assertRaisesRegex(ProvenanceError, "unstaged tracked changes"):
                source_identity(root)
            subprocess.run(["git", "-C", str(root), "checkout", "--", "main.py"], check=True)

            (root / "extra.py").write_text("pass\n", encoding="utf-8")
            with self.assertRaisesRegex(ProvenanceError, "untracked source"):
                source_identity(root)

    def test_git_mode_cannot_downgrade_to_archive_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
            subprocess.run(
                ["git", "-C", str(root), "config", "user.email", "test@example.invalid"],
                check=True,
            )
            (root / "main.py").write_text("pass\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "main.py"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "initial"], check=True)
            (root / REVISION_NAME).write_text(REVISION + "\n", encoding="ascii")
            (root / MANIFEST_NAME).write_text("{}\n", encoding="utf-8")
            identity = source_identity(root)
            self.assertEqual(identity.mode, "git")
            self.assertNotEqual(identity.revision, REVISION)

    def test_broken_git_metadata_cannot_fall_back_to_archive_mode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / ".git").mkdir()
            (root / "main.py").write_text("pass\n", encoding="utf-8")
            create_archive_manifest(root, REVISION)
            with self.assertRaisesRegex(ProvenanceError, "Git metadata exists"):
                source_identity(root)


if __name__ == "__main__":
    unittest.main()
