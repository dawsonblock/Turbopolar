"""Tests for _compute_git_dirty_hash().

These tests use real temporary Git repositories to verify deterministic
dirty-state hashing across tracked modifications, staged changes, untracked
files, nested directories, and edge cases.
"""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from rfsn_v11.promotion.provenance import _compute_git_dirty_hash


class TestGitDirtyHash(unittest.TestCase):
    """P0: Verify dirty-tree provenance hashing with real Git repos."""

    def setUp(self):
        self.original_cwd = os.getcwd()
        self.tmpdir = tempfile.mkdtemp()
        self._run_in_tmp(["git", "init"])
        self._run_in_tmp(["git", "config", "user.email", "test@test.com"])
        self._run_in_tmp(["git", "config", "user.name", "Test"])
        # Create an initial tracked file and commit it
        (Path(self.tmpdir) / "tracked.txt").write_text("initial")
        self._run_in_tmp(["git", "add", "tracked.txt"])
        self._run_in_tmp(["git", "commit", "-m", "initial"])
        os.chdir(self.tmpdir)

    def tearDown(self):
        os.chdir(self.original_cwd)
        shutil.rmtree(self.tmpdir)

    def _run_in_tmp(self, cmd):
        subprocess.run(cmd, cwd=self.tmpdir, check=True, capture_output=True)

    def test_clean_tree_returns_empty_string(self):
        """A clean working tree should produce an empty dirty hash."""
        result = _compute_git_dirty_hash()
        self.assertEqual(result, "")

    def test_modified_tracked_file_changes_hash(self):
        """Modifying a tracked file must produce a non-empty hash."""
        (Path(self.tmpdir) / "tracked.txt").write_text("modified")
        h1 = _compute_git_dirty_hash()
        self.assertNotEqual(h1, "")
        self.assertEqual(len(h1), 64)

        # Same modification should yield identical hash
        h2 = _compute_git_dirty_hash()
        self.assertEqual(h1, h2)

        # Different modification should yield different hash
        (Path(self.tmpdir) / "tracked.txt").write_text("different")
        h3 = _compute_git_dirty_hash()
        self.assertNotEqual(h1, h3)

    def test_staged_file_changes_hash(self):
        """Staging a change must produce a deterministic hash."""
        (Path(self.tmpdir) / "tracked.txt").write_text("staged")
        self._run_in_tmp(["git", "add", "tracked.txt"])
        h1 = _compute_git_dirty_hash()
        self.assertNotEqual(h1, "")
        h2 = _compute_git_dirty_hash()
        self.assertEqual(h1, h2)

    def test_single_untracked_file(self):
        """One untracked file must be included in the dirty hash."""
        (Path(self.tmpdir) / "untracked.txt").write_text("secret")
        h1 = _compute_git_dirty_hash()
        self.assertNotEqual(h1, "")
        self.assertEqual(len(h1), 64)

        # Re-running must be deterministic
        h2 = _compute_git_dirty_hash()
        self.assertEqual(h1, h2)

    def test_untracked_content_change_changes_hash(self):
        """Changing the content of an untracked file must change the hash."""
        (Path(self.tmpdir) / "untracked.txt").write_text("before")
        h1 = _compute_git_dirty_hash()
        (Path(self.tmpdir) / "untracked.txt").write_text("after")
        h2 = _compute_git_dirty_hash()
        self.assertNotEqual(h1, h2)

    def test_nested_untracked_directory(self):
        """Untracked directories must be hashed recursively."""
        nested = Path(self.tmpdir) / "nested" / "deep"
        nested.mkdir(parents=True)
        (nested / "file.txt").write_text("deep content")
        h1 = _compute_git_dirty_hash()
        self.assertNotEqual(h1, "")
        self.assertEqual(len(h1), 64)

        # Re-running must be deterministic
        h2 = _compute_git_dirty_hash()
        self.assertEqual(h1, h2)

    def test_nested_content_change_changes_hash(self):
        """Changing content inside a nested untracked directory must
        change hash."""
        nested = Path(self.tmpdir) / "nested" / "deep"
        nested.mkdir(parents=True)
        (nested / "file.txt").write_text("original")
        h1 = _compute_git_dirty_hash()
        (nested / "file.txt").write_text("changed")
        h2 = _compute_git_dirty_hash()
        self.assertNotEqual(h1, h2)

    def test_unreadable_file_raises(self):
        """An unreadable untracked file must raise RuntimeError."""
        bad = Path(self.tmpdir) / "unreadable.txt"
        bad.write_text("data")
        bad.chmod(0o000)
        try:
            # Skip if running as a privileged user where chmod 000 does not
            # prevent reading (e.g., root on Linux).
            try:
                bad.read_text()
                raise unittest.SkipTest(
                    "Running with elevated privileges; chmod 000 does not "
                    "prevent file reads in this environment."
                )
            except OSError:
                pass
            with self.assertRaises(RuntimeError):
                _compute_git_dirty_hash()
        finally:
            bad.chmod(0o644)

    def test_path_with_spaces(self):
        """Untracked paths containing spaces must be handled correctly."""
        spaced = Path(self.tmpdir) / "path with spaces" / "file.txt"
        spaced.parent.mkdir(parents=True)
        spaced.write_text("spaced content")
        h1 = _compute_git_dirty_hash()
        self.assertNotEqual(h1, "")
        self.assertEqual(len(h1), 64)

        # Must be deterministic
        h2 = _compute_git_dirty_hash()
        self.assertEqual(h1, h2)

    def test_tracked_and_untracked_together(self):
        """Both tracked modifications and untracked files must contribute."""
        (Path(self.tmpdir) / "tracked.txt").write_text("modified")
        (Path(self.tmpdir) / "new.txt").write_text("untracked")
        h = _compute_git_dirty_hash()
        self.assertNotEqual(h, "")
        self.assertEqual(len(h), 64)


if __name__ == "__main__":
    unittest.main()
