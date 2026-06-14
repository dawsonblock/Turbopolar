"""Tests for canonical exact token fixture schema."""

import unittest
import tempfile
from pathlib import Path

from benchmarks.prompt_fixtures import (
    ExactTokenFixture,
    build_exact_token_fixtures,
    validate_fixture_hashes,
    write_token_fixtures,
    load_token_fixtures_canonical,
)


class TestExactTokenFixture(unittest.TestCase):
    """Test canonical fixture schema and validation."""

    def test_fixture_creation(self):
        """ExactTokenFixture should create with all required fields."""
        fixture = ExactTokenFixture(
            fixture_id="short_16",
            category="short",
            length=16,
            tokens=tuple(range(10, 26)),
            content_hash="abc123",
        )
        self.assertEqual(fixture.fixture_id, "short_16")
        self.assertEqual(fixture.length, 16)
        self.assertEqual(len(fixture.tokens), 16)

    def test_from_dict_without_hash(self):
        """from_dict should compute content hash if not provided."""
        data = {
            "category": "short",
            "length": 16,
            "tokens": list(range(10, 26)),
        }
        fixture = ExactTokenFixture.from_dict(data)
        self.assertIsNotNone(fixture.content_hash)
        self.assertEqual(fixture.fixture_id, "short_16")
        self.assertEqual(fixture.length, 16)

    def test_from_dict_with_hash(self):
        """from_dict should use provided content hash."""
        expected_hash = "provided_hash_123"
        data = {
            "category": "short",
            "length": 16,
            "tokens": list(range(10, 26)),
            "content_hash": expected_hash,
        }
        fixture = ExactTokenFixture.from_dict(data)
        self.assertEqual(fixture.content_hash, expected_hash)

    def test_compute_content_hash_deterministic(self):
        """Content hash computation should be deterministic."""
        tokens = tuple(range(10, 26))
        hash1 = ExactTokenFixture._compute_content_hash(tokens)
        hash2 = ExactTokenFixture._compute_content_hash(tokens)
        self.assertEqual(hash1, hash2)

    def test_compute_fixture_id_deterministic(self):
        """Fixture ID computation should be deterministic."""
        id1 = ExactTokenFixture._compute_fixture_id("short", 16)
        id2 = ExactTokenFixture._compute_fixture_id("short", 16)
        self.assertEqual(id1, id2)
        self.assertEqual(id1, "short_16")

    def test_build_exact_token_fixtures(self):
        """build_exact_token_fixtures should return canonical fixtures."""
        fixtures = build_exact_token_fixtures(
            specs=[("short", 16), ("medium", 32)]
        )
        self.assertEqual(len(fixtures), 2)
        self.assertIsInstance(fixtures[0], ExactTokenFixture)
        self.assertEqual(fixtures[0].fixture_id, "short_16")
        self.assertEqual(fixtures[1].fixture_id, "medium_32")
        self.assertIsNotNone(fixtures[0].content_hash)
        self.assertIsNotNone(fixtures[1].content_hash)

    def test_validate_fixture_hashes_passing(self):
        """validate_fixture_hashes should pass for matching hashes."""
        fixtures = build_exact_token_fixtures(specs=[("short", 16)])
        expected_hashes = {f.fixture_id: f.content_hash for f in fixtures}
        errors = validate_fixture_hashes(fixtures, expected_hashes)
        self.assertEqual(errors, [])

    def test_validate_fixture_hashes_mismatch(self):
        """validate_fixture_hashes should detect hash mismatches."""
        fixtures = build_exact_token_fixtures(specs=[("short", 16)])
        expected_hashes = {fixtures[0].fixture_id: "wrong_hash"}
        errors = validate_fixture_hashes(fixtures, expected_hashes)
        self.assertTrue(len(errors) > 0)
        self.assertTrue("hash mismatch" in errors[0])

    def test_validate_fixture_hashes_missing(self):
        """validate_fixture_hashes should detect missing expected hashes."""
        fixtures = build_exact_token_fixtures(specs=[("short", 16)])
        expected_hashes = {}  # Empty
        errors = validate_fixture_hashes(fixtures, expected_hashes)
        self.assertTrue(len(errors) > 0)
        self.assertTrue("no expected hash" in errors[0])

    def test_write_and_load_canonical_fixtures(self):
        """write_token_fixtures and load_token_fixtures_canonical should round-trip."""
        fixtures = build_exact_token_fixtures(specs=[("short", 16), ("medium", 32)])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "fixtures.jsonl"
            write_token_fixtures(fixtures, path)
            
            loaded = load_token_fixtures_canonical(path)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[0].fixture_id, fixtures[0].fixture_id)
            self.assertEqual(loaded[0].content_hash, fixtures[0].content_hash)
            self.assertEqual(loaded[1].fixture_id, fixtures[1].fixture_id)
            self.assertEqual(loaded[1].content_hash, fixtures[1].content_hash)

    def test_load_canonical_validates_hash_mismatch(self):
        """load_token_fixtures_canonical should detect hash mismatches."""
        fixtures = build_exact_token_fixtures(specs=[("short", 16)])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "fixtures.jsonl"
            # Write with wrong hash
            import json
            with open(path, "w") as f:
                fixture_dict = {
                    "fixture_id": fixtures[0].fixture_id,
                    "category": fixtures[0].category,
                    "length": fixtures[0].length,
                    "tokens": list(fixtures[0].tokens),
                    "content_hash": "wrong_hash",
                }
                f.write(json.dumps(fixture_dict) + "\n")
            
            with self.assertRaises(ValueError) as context:
                load_token_fixtures_canonical(path)
            self.assertIn("content hash mismatch", str(context.exception))

    def test_load_canonical_validates_vocab_size(self):
        """load_token_fixtures_canonical should validate token IDs against vocab size."""
        fixtures = build_exact_token_fixtures(specs=[("short", 16)])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "fixtures.jsonl"
            write_token_fixtures(fixtures, path)
            
            # Use a vocab size smaller than the max token ID
            with self.assertRaises(ValueError) as context:
                load_token_fixtures_canonical(path, vocab_size=5)
            self.assertIn("exceeds vocabulary size", str(context.exception))


if __name__ == "__main__":
    unittest.main()