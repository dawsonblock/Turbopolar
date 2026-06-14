#!/usr/bin/env python3
"""Generate extended token fixtures for required fused-decode context lengths.

The fused decode benchmark needs context + 129 continuation tokens:
- 512 + 129 = 641 tokens
- 2048 + 129 = 2177 tokens
- 4096 + 129 = 4225 tokens
- 8192 + 129 = 8321 tokens
- 16384 + 129 = 16513 tokens
"""

import hashlib
import json
from pathlib import Path

FIXTURES_PATH = Path(__file__).parent / "exact_token_fixtures.jsonl"


def generate_deterministic_tokens(length: int, seed: int = 42) -> list:
    """Generate deterministic token sequences using a simple LCG."""
    tokens = []
    state = seed
    for i in range(length):
        # Simple linear congruential generator
        state = (1664525 * state + 1013904223) % (2 ** 32)
        # Map to reasonable token range (10-10000 to avoid special tokens)
        token = 10 + (state % 9990)
        tokens.append(token)
    return tokens


def main():
    # Required lengths for fused decode (context + 129 continuation)
    required_lengths = {
        "fused_512": 641,      # 512 context + 129 continuation
        "fused_2048": 2177,    # 2048 context + 129 continuation
        "fused_4096": 4225,    # 4096 context + 129 continuation
        "fused_8192": 8321,    # 8192 context + 129 continuation
        "fused_16384": 16513,  # 16384 context + 129 continuation
    }

    # Read existing fixtures
    existing_fixtures = []
    if FIXTURES_PATH.exists():
        with open(FIXTURES_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    existing_fixtures.append(json.loads(line))

    # Generate fixtures, replacing existing ones by category
    existing_by_category = {f["category"]: f for f in existing_fixtures}
    all_fixtures = []
    
    for category, length in required_lengths.items():
        # Use SHA-256 based seed for cross-process determinism
        seed = int.from_bytes(
            hashlib.sha256(category.encode()).digest()[:4],
            "big",
        )
        tokens = generate_deterministic_tokens(length, seed)

        fixture = {
            "category": category,
            "length": length,
            "tokens": tokens,
        }
        all_fixtures.append(fixture)
        print(f"Generated {category}: {length} tokens")
    
    # Add any existing fixtures that are not in required_lengths
    for existing_fixture in existing_fixtures:
        if existing_fixture["category"] not in required_lengths:
            all_fixtures.append(existing_fixture)

    # Write back
    with open(FIXTURES_PATH, "w") as f:
        for fixture in all_fixtures:
            f.write(json.dumps(fixture) + "\n")

    print(f"\nWrote {len(all_fixtures)} fixtures to {FIXTURES_PATH}")
    print(f"Generated {len(required_lengths)} fixtures, preserved {len(all_fixtures) - len(required_lengths)} existing fixtures")


if __name__ == "__main__":
    main()