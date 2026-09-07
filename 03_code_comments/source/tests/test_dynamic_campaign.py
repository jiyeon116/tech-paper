from __future__ import annotations

import argparse
import importlib.util
import json
import os
import stat
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

from dynamic_experiment.provenance import create_archive_manifest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_dynamic_campaign.py"
REPO_ROOT = MODULE_PATH.parents[1]
TEST_REVISION = "a" * 40
SPEC = importlib.util.spec_from_file_location("dynamic_campaign_under_test", MODULE_PATH)
campaign = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = campaign
SPEC.loader.exec_module(campaign)


FAKE_RUNNER = """#!__PYTHON__
import argparse
import json
import os
import sys
import time
import hashlib
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--config')
parser.add_argument('--seed', type=int)
parser.add_argument('--output-dir', type=Path)
args = parser.parse_args(sys.argv[3:])
mode = os.environ.get('FAKE_DYNAMIC_MODE', 'complete')
if mode == 'sleep':
    time.sleep(30)
elif mode == 'fail':
    raise SystemExit(7)
else:
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary = {
        'status': 'complete',
        'config_hash': os.environ['QECO_CAMPAIGN_CONFIG_HASH'],
        'seed': args.seed,
        'revision': os.environ['QECO_CAMPAIGN_REVISION'],
        'source_identity_hash': os.environ['QECO_SOURCE_IDENTITY_HASH'],
        'metrics': {'qoe': 1.0},
    }
    (args.output_dir / 'summary.json').write_text(json.dumps(summary), encoding='utf-8')
    digest = hashlib.sha256((args.output_dir / 'summary.json').read_bytes()).hexdigest()
    (args.output_dir / 'checksums.json').write_text(
        json.dumps({'summary.json': digest}), encoding='utf-8')
"""


class DynamicCampaignTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = self.root / "config.json"
        self.config.write_text('{}\n', encoding="utf-8")
        self.python = self.root / "fake_python"
        self.python.write_text(
            textwrap.dedent(FAKE_RUNNER).replace("__PYTHON__", sys.executable), encoding="utf-8"
        )
        self.python.chmod(self.python.stat().st_mode | stat.S_IXUSR)
        self.output = self.root / "results"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def args(self, **overrides: object) -> argparse.Namespace:
        values = {
            "config": self.config,
            "output_dir": self.output,
            "python": self.python,
            "seeds": [42],
            "budget_seconds": 5.0,
            "cell_timeout_seconds": 2.0,
            "min_free_gib": 0.0,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def environment(self, mode: str = "complete"):
        return mock.patch.dict(
            os.environ,
            {"FAKE_DYNAMIC_MODE": mode},
            clear=False,
        )

    def identity(self):
        return campaign.SourceIdentity("git", TEST_REVISION, "b" * 64, 20)

    @mock.patch.object(campaign, "source_identity")
    def test_completed_cell_is_reused_only_with_matching_identity_and_checksum(self, identity):
        identity.return_value = self.identity()
        with self.environment():
            self.assertEqual(campaign.run_campaign(self.args(), poll_seconds=0.01), 0)
            self.assertEqual(campaign.run_campaign(self.args(), poll_seconds=0.01), 0)

        attempts = sorted((self.output / "seed_42").glob("attempt_*"))
        self.assertEqual([path.name for path in attempts], ["attempt_001"])
        manifest = json.loads((self.output / campaign.MANIFEST_NAME).read_text())
        self.assertEqual(manifest["status"], "complete")
        self.assertTrue(manifest["seeds"]["42"]["reused"])

    @mock.patch.object(campaign, "source_identity")
    def test_mismatched_completed_summary_starts_new_attempt_without_overwrite(self, identity):
        identity.return_value = self.identity()
        with self.environment():
            self.assertEqual(campaign.run_campaign(self.args(), poll_seconds=0.01), 0)

        first = self.output / "seed_42" / "attempt_001"
        summary_path = first / "result" / "summary.json"
        checksums_path = first / "result" / "checksums.json"
        summary = json.loads(summary_path.read_text())
        summary["config_hash"] = "wrong"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        checksums_path.write_text(
            json.dumps({"summary.json": campaign.sha256_file(summary_path)}), encoding="utf-8"
        )

        with self.environment():
            self.assertEqual(campaign.run_campaign(self.args(), poll_seconds=0.01), 0)

        attempts = sorted((self.output / "seed_42").glob("attempt_*"))
        self.assertEqual([path.name for path in attempts], ["attempt_001", "attempt_002"])
        self.assertEqual(json.loads(summary_path.read_text())["config_hash"], "wrong")

    @mock.patch.object(campaign, "source_identity")
    def test_changed_source_identity_cannot_reuse_completed_attempt(self, identity):
        identity.side_effect = [
            self.identity(),
            campaign.SourceIdentity("git", TEST_REVISION, "c" * 64, 20),
        ]
        with self.environment():
            self.assertEqual(campaign.run_campaign(self.args(), poll_seconds=0.01), 0)
            self.assertEqual(campaign.run_campaign(self.args(), poll_seconds=0.01), 0)

        attempts = sorted((self.output / "seed_42").glob("attempt_*"))
        self.assertEqual([path.name for path in attempts], ["attempt_001", "attempt_002"])

    @mock.patch.object(campaign, "source_identity")
    def test_checksum_omitting_result_artifact_forces_new_attempt(self, identity):
        identity.return_value = self.identity()
        with self.environment():
            self.assertEqual(campaign.run_campaign(self.args(), poll_seconds=0.01), 0)
        first_result = self.output / "seed_42" / "attempt_001" / "result"
        (first_result / "episodes.jsonl").write_text("{}\n", encoding="utf-8")

        with self.environment():
            self.assertEqual(campaign.run_campaign(self.args(), poll_seconds=0.01), 0)

        attempts = sorted((self.output / "seed_42").glob("attempt_*"))
        self.assertEqual([path.name for path in attempts], ["attempt_001", "attempt_002"])

    @mock.patch.object(campaign, "source_identity")
    def test_malformed_existing_campaign_manifest_is_preserved_and_aborts(self, identity):
        identity.return_value = self.identity()
        self.output.mkdir()
        manifest = self.output / campaign.MANIFEST_NAME
        original = b"{not valid json\n"
        manifest.write_bytes(original)

        with self.assertRaisesRegex(SystemExit, "preserving it unchanged"):
            campaign.run_campaign(self.args(), poll_seconds=0.01)

        self.assertEqual(manifest.read_bytes(), original)
        self.assertFalse((self.output / "seed_42").exists())

    @mock.patch.object(campaign, "source_identity")
    def test_timeout_kills_owned_child_and_preserves_attempt(self, identity):
        identity.return_value = self.identity()
        with self.environment("sleep"):
            result = campaign.run_campaign(
                self.args(cell_timeout_seconds=0.1, budget_seconds=1.0), poll_seconds=0.01
            )

        self.assertEqual(result, 1)
        attempt = self.output / "seed_42" / "attempt_001"
        status = json.loads((attempt / campaign.STATUS_NAME).read_text())
        self.assertEqual(status["state"], "timeout")
        self.assertIn("[EXIT STATUS]", (attempt / "run.log").read_text())

    @mock.patch.object(campaign, "source_identity")
    def test_failed_cell_restarts_in_new_attempt_on_next_campaign_run(self, identity):
        identity.return_value = self.identity()
        with self.environment("fail"):
            self.assertEqual(campaign.run_campaign(self.args(), poll_seconds=0.01), 1)
        first = self.output / "seed_42" / "attempt_001"
        self.assertEqual(json.loads((first / campaign.STATUS_NAME).read_text())["state"], "failed")

        with self.environment():
            self.assertEqual(campaign.run_campaign(self.args(), poll_seconds=0.01), 0)

        self.assertTrue(first.is_dir())
        second_summary = self.output / "seed_42" / "attempt_002" / "result" / "summary.json"
        self.assertTrue(second_summary.is_file())

    def test_next_attempt_ignores_unrelated_directories(self):
        seed_dir = self.output / "seed_42"
        (seed_dir / "attempt_001").mkdir(parents=True)
        (seed_dir / "attempt_bad").mkdir()
        (seed_dir / "notes").mkdir()
        self.assertEqual(campaign.next_attempt(seed_dir).name, "attempt_002")

    def test_direct_file_cli_finds_repo_package_from_unrelated_working_directory(self):
        archive = self.root / "archive"
        (archive / "scripts").mkdir(parents=True)
        (archive / "dynamic_experiment").mkdir()
        for relative in (
            "scripts/run_dynamic_campaign.py",
            "dynamic_experiment/__init__.py",
            "dynamic_experiment/config.py",
            "dynamic_experiment/provenance.py",
        ):
            destination = archive / relative
            destination.write_bytes((REPO_ROOT / relative).read_bytes())
        create_archive_manifest(archive, TEST_REVISION)
        output = self.root / "cli-results"
        environment = os.environ.copy()
        environment["FAKE_DYNAMIC_MODE"] = "complete"
        completed = subprocess.run(
            [
                sys.executable,
                str(archive / "scripts" / "run_dynamic_campaign.py"),
                "--config",
                str(self.config),
                "--output-dir",
                str(output),
                "--python",
                str(self.python),
                "--seeds",
                "42",
                "--budget-seconds",
                "5",
                "--cell-timeout-seconds",
                "2",
                "--min-free-gib",
                "0",
            ],
            cwd=self.root,
            env=environment,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
        self.assertTrue((output / "seed_42" / "attempt_001" / "result" / "summary.json").is_file())

    @mock.patch.object(campaign, "source_identity")
    def test_python_symlink_path_is_preserved_in_manifest_and_child_command(self, identity):
        identity.return_value = self.identity()
        linked_python = self.root / "env_channel" / "bin" / "python"
        linked_python.parent.mkdir(parents=True)
        linked_python.symlink_to(self.python)
        args = self.args(python=linked_python)

        with self.environment():
            self.assertEqual(campaign.run_campaign(args, poll_seconds=0.01), 0)

        manifest = json.loads((self.output / campaign.MANIFEST_NAME).read_text())
        status = json.loads(
            (self.output / "seed_42" / "attempt_001" / campaign.STATUS_NAME).read_text()
        )
        self.assertEqual(manifest["python"], str(linked_python))
        self.assertEqual(status["command"][0], str(linked_python))

    def test_cli_rejects_non_finite_time_and_disk_values(self):
        with self.assertRaises(SystemExit):
            campaign.parse_args(
                [
                    "--config", str(self.config), "--output-dir", str(self.output),
                    "--python", str(self.python), "--seeds", "42",
                    "--budget-seconds", "nan",
                ]
            )
        with self.assertRaises(SystemExit):
            campaign.parse_args(
                [
                    "--config", str(self.config), "--output-dir", str(self.output),
                    "--python", str(self.python), "--seeds", "42",
                    "--min-free-gib", "inf",
                ]
            )


if __name__ == "__main__":
    unittest.main()
