# Ported from Project Asma (https://github.com/Foundation-0/project-asma), 2026-11-13.
# Copyright (c) 2026 Project Asma contributors. MIT licensed.
# See docs/verification-tooling.md for provenance and local changes.

"""Tests for registry implementation hash verification."""

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import check_implementation_hashes as check_module
from check_implementation_hashes import (
    STATUS_OK,
    STATUS_STALE,
    STATUS_UNKNOWN,
    check_entry,
    check_implementation_hashes,
    failures,
    file_sha256,
    is_inside_git_work_tree,
)

ROOT = Path(__file__).resolve().parents[1]
GIT_WORK_TREE = is_inside_git_work_tree(ROOT)
SKIP_WITHOUT_GIT = "not inside a git work tree (stale-hash detection needs history)"


class GitWorkTreeDetection(unittest.TestCase):
    @unittest.skipUnless(shutil.which("git"), "git is not installed")
    def test_nested_tree_in_foreign_git_repo_is_not_checkout(self):
        with tempfile.TemporaryDirectory() as tmp:
            outer = Path(tmp) / "foreign"
            outer.mkdir()
            subprocess.run(["git", "init", "-q"], cwd=outer, check=True)
            subprocess.run(
                ["git", "config", "user.email", "test@example.com"],
                cwd=outer,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "test"],
                cwd=outer,
                check=True,
            )
            nested = outer / "project-asma"
            shutil.copytree(
                ROOT,
                nested,
                ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"),
            )
            self.assertFalse(is_inside_git_work_tree(nested))


class ImplementationHashChecks(unittest.TestCase):
    def test_current_tree_passes(self):
        self.assertEqual(failures(check_implementation_hashes(ROOT)), [])

    def test_unknown_hash_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "registries").mkdir()
            (root / "tools").mkdir()
            (root / "tools" / "example.py").write_text("v1\n", encoding="utf-8")
            payload = {
                "schema_version": 1,
                "implementation_sha256": {"example.py": "0" * 64},
            }
            (root / "registries" / "pilot-001-example.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            (root / ".git").mkdir()
            result = check_entry(
                root,
                "pilot-001-example.json",
                payload,
                "example.py",
                "0" * 64,
            )
            self.assertEqual(result.status, STATUS_UNKNOWN)

    def test_current_hash_is_ok(self):
        tool = ROOT / "tools" / "evaluate_heldout.py"
        current = file_sha256(tool)
        data = {"implementation_sha256": {"evaluate_heldout.py": current}}
        result = check_entry(
            ROOT,
            "ported-current.json",
            data,
            "evaluate_heldout.py",
            current,
        )
        self.assertEqual(result.status, STATUS_OK)


if __name__ == "__main__":
    unittest.main()
