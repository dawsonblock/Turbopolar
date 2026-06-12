"""Tests for deterministic prompt fixture helpers."""

import unittest
from pathlib import Path

from benchmarks.prompt_fixtures import (
    build_exact_token_fixtures,
    load_token_fixtures,
    normalize_prompts,
    write_token_fixtures,
)


class _FakeTokenizer:
    def encode(self, text: str):
        return [ord(c) % 100 for c in text]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)


class TestPromptFixtures(unittest.TestCase):
    def test_build_exact_token_fixtures_lengths(self):
        fixtures = build_exact_token_fixtures()
        self.assertEqual(len(fixtures), len(set(f["length"] for f in fixtures)))
        for fx in fixtures:
            self.assertEqual(len(fx["tokens"]), fx["length"])
            self.assertIn(fx["category"], ("short", "boundary", "medium", "long", "stress"))

    def test_token_fixtures_are_deterministic(self):
        a = build_exact_token_fixtures()
        b = build_exact_token_fixtures()
        self.assertEqual(a, b)

    def test_write_and_load_token_fixtures(self):
        fixtures = build_exact_token_fixtures(specs=[("short", 8), ("boundary", 16)])
        path = Path("/tmp/test_turbopolar_fixtures.jsonl")
        write_token_fixtures(fixtures, path)
        loaded = load_token_fixtures(path)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0]["length"], 8)
        self.assertEqual(loaded[1]["length"], 16)
        path.unlink()

    def test_normalize_text_prompts_returns_tokens(self):
        suite_path = Path(__file__).resolve().parents[2] / "benchmarks" / "prompt_suite.jsonl"
        normalized = normalize_prompts(_FakeTokenizer(), suite_path)
        self.assertTrue(len(normalized) > 0)
        for entry in normalized:
            self.assertIn("category", entry)
            self.assertIn("tokens", entry)
            self.assertIn("text", entry)
            self.assertIsInstance(entry["tokens"], list)

    def test_normalize_token_fixtures_returns_exact_lengths(self):
        fixtures_path = (
            Path(__file__).resolve().parents[2] / "benchmarks" / "exact_token_fixtures.jsonl"
        )
        normalized = normalize_prompts(_FakeTokenizer(), fixtures_path)
        self.assertTrue(len(normalized) > 0)
        for entry in normalized:
            self.assertIn(entry["category"], ("short", "boundary", "medium", "long", "stress"))
            self.assertEqual(len(entry["tokens"]), len(_FakeTokenizer().encode(entry["text"])))


if __name__ == "__main__":
    unittest.main()
