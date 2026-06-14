"""Deterministic, exact-length prompt fixtures for real-model benchmarks.

The fixtures in ``exact_token_fixtures.jsonl`` store explicit token-id sequences
so that prompt lengths are independent of tokenizer merging behavior.  This makes
perplexity, compression, and decode-speed comparisons reproducible across models.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ExactTokenFixture:
    """Canonical schema for exact token fixtures.

    Provides deterministic fixture IDs and content hashes for reproducibility.
    """

    fixture_id: str
    category: str
    length: int
    tokens: Tuple[int, ...]
    content_hash: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExactTokenFixture":
        """Create fixture from dictionary, computing content hash if needed."""
        tokens = tuple(int(t) for t in data.get("tokens", []))
        content_hash = data.get("content_hash")
        if content_hash is None:
            content_hash = cls._compute_content_hash(tokens)
        
        fixture_id = data.get("fixture_id")
        if fixture_id is None:
            fixture_id = cls._compute_fixture_id(
                data.get("category", "unknown"),
                len(tokens)
            )
        
        return cls(
            fixture_id=fixture_id,
            category=data.get("category", "unknown"),
            length=len(tokens),
            tokens=tokens,
            content_hash=content_hash,
        )
    
    @staticmethod
    def _compute_content_hash(tokens: Tuple[int, ...]) -> str:
        """Compute SHA-256 hash of token sequence."""
        content = json.dumps(list(tokens), sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
    
    @staticmethod
    def _compute_fixture_id(category: str, length: int) -> str:
        """Compute deterministic fixture ID from category and length."""
        return f"{category}_{length}"


CATEGORY_LENGTHS: List[Tuple[str, int]] = [
    ("short", 16),
    ("short", 32),
    ("boundary", 64),
    ("medium", 128),
    ("medium", 256),
    ("long", 512),
    ("long", 1024),
    ("stress", 2048),
    ("stress", 4096),
    ("stress", 8192),  # P0: Add for 8K context + 129 tokens
    ("stress", 16384),  # P0: Add for 16K context + 129 tokens
    # Fused-decode specific fixtures (context + 129 continuation tokens)
    ("fused_512", 641),      # 512 context + 129 continuation
    ("fused_2048", 2177),    # 2048 context + 129 continuation
    ("fused_4096", 4225),    # 4096 context + 129 continuation
    ("fused_8192", 8321),    # 8192 context + 129 continuation
    ("fused_16384", 16513),  # 16384 context + 129 continuation
]


def _token_sequence(
    length: int, start_id: int = 10, vocab_stride: int = 80
) -> List[int]:
    """Return a deterministic token-id sequence of the requested length."""
    return [(start_id + (i % vocab_stride)) for i in range(length)]


def build_exact_token_fixtures(
    specs: Optional[List[Tuple[str, int]]] = None,
    start_id: int = 10,
    vocab_stride: int = 80,
) -> List[ExactTokenFixture]:
    """Build a deterministic prompt fixture list with exact token lengths.

    Args:
        specs: list of (category, length) pairs.  Defaults to a short-to-stress
            progression aligned with the 64-token TurboPolar block size.
        start_id: smallest token id to use; must be < model vocab size.
        vocab_stride: number of distinct token ids to cycle through.

    Returns:
        List of ExactTokenFixture objects with deterministic IDs and content hashes.
    """
    specs = specs or CATEGORY_LENGTHS
    fixtures = []
    for category, length in specs:
        tokens = _token_sequence(length, start_id=start_id, vocab_stride=vocab_stride)
        fixture_id = ExactTokenFixture._compute_fixture_id(category, length)
        content_hash = ExactTokenFixture._compute_content_hash(tuple(tokens))
        fixtures.append(ExactTokenFixture(
            fixture_id=fixture_id,
            category=category,
            length=length,
            tokens=tuple(tokens),
            content_hash=content_hash,
        ))
    return fixtures


def validate_fixture_hashes(
    fixtures: List[ExactTokenFixture],
    expected_hashes: Dict[str, str],
) -> List[str]:
    """Validate that fixture content hashes match expected values.

    Args:
        fixtures: List of ExactTokenFixture objects to validate.
        expected_hashes: Dictionary mapping fixture_id to expected content_hash.

    Returns:
        List of error messages. Empty list means all fixtures are valid.
    """
    errors = []
    for fixture in fixtures:
        expected_hash = expected_hashes.get(fixture.fixture_id)
        if expected_hash is None:
            errors.append(f"Fixture {fixture.fixture_id} has no expected hash")
        elif fixture.content_hash != expected_hash:
            errors.append(
                f"Fixture {fixture.fixture_id} hash mismatch: "
                f"expected {expected_hash}, got {fixture.content_hash}"
            )
    return errors


def write_token_fixtures(
    fixtures: List[ExactTokenFixture],
    path: Path,
) -> Path:
    """Write fixture dicts to a JSONL file, one fixture per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for fixture in fixtures:
            fixture_dict = {
                "fixture_id": fixture.fixture_id,
                "category": fixture.category,
                "length": fixture.length,
                "tokens": list(fixture.tokens),
                "content_hash": fixture.content_hash,
            }
            f.write(json.dumps(fixture_dict) + "\n")
    return path


def load_token_fixtures_canonical(path: Path, vocab_size: Optional[int] = None) -> List[ExactTokenFixture]:
    """Load exact-token fixtures from a JSONL file using canonical schema.
    
    Args:
        path: Path to the JSONL fixture file.
        vocab_size: Optional vocabulary size for validation. If provided, validates
            that all token IDs are within the valid range [0, vocab_size).
    
    Returns:
        List of ExactTokenFixture objects.
    
    Raises:
        ValueError: If vocab_size is provided and any token ID exceeds vocab_size - 1,
            or if fixture content hash doesn't match computed hash.
    """
    fixtures = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            fixture = ExactTokenFixture.from_dict(obj)
            
            # Validate token IDs against vocabulary size
            if vocab_size is not None:
                max_token = max(fixture.tokens) if fixture.tokens else 0
                if max_token >= vocab_size:
                    raise ValueError(
                        f"Fixture {fixture.fixture_id} contains token ID {max_token} "
                        f"which exceeds vocabulary size {vocab_size}"
                    )
            
            # Validate content hash matches
            computed_hash = ExactTokenFixture._compute_content_hash(fixture.tokens)
            if fixture.content_hash != computed_hash:
                raise ValueError(
                    f"Fixture {fixture.fixture_id} content hash mismatch: "
                    f"stored {fixture.content_hash}, computed {computed_hash}"
                )
            
            fixtures.append(fixture)
    return fixtures


def load_text_prompts(path: Path) -> List[str]:
    """Load plain-text prompts from a JSONL file (legacy text suite)."""
    prompts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                prompts.append(obj.get("prompt", obj.get("text", "")))
            elif isinstance(obj, str):
                prompts.append(obj)
    return prompts


def load_token_fixtures(path: Path, vocab_size: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load exact-token fixtures from a JSONL file.
    
    Args:
        path: Path to the JSONL fixture file.
        vocab_size: Optional vocabulary size for validation. If provided, validates
            that all token IDs are within the valid range [0, vocab_size).
    
    Returns:
        List of fixture dicts with category, length, and tokens keys.
    
    Raises:
        ValueError: If vocab_size is provided and any token ID exceeds vocab_size - 1.
    """
    fixtures = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict) and "tokens" in obj:
                tokens = [int(t) for t in obj["tokens"]]
                
                # Validate token IDs against vocabulary size
                if vocab_size is not None:
                    max_token = max(tokens) if tokens else 0
                    if max_token >= vocab_size:
                        raise ValueError(
                            f"Fixture {obj.get('category', 'unknown')} contains token ID {max_token} "
                            f"which exceeds vocabulary size {vocab_size}"
                        )
                
                fixtures.append(obj)
    return fixtures


def normalize_prompts(
    tokenizer,
    source: Path,
    vocab_size: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load prompts from *source* and normalize each entry to ``{category, tokens, text}``.

    Text entries (legacy) are encoded with ``tokenizer``; token fixtures are used
    verbatim.  The first fixture's category is ``default`` for plain text entries.
    
    Args:
        tokenizer: Tokenizer with encode/decode methods.
        source: Path to JSONL file with prompts or token fixtures.
        vocab_size: Optional vocabulary size for validation. If provided, validates
            that all token IDs are within the valid range [0, vocab_size).
    
    Raises:
        ValueError: If vocab_size is provided and any token ID exceeds vocab_size - 1.
    """
    normalized: List[Dict[str, Any]] = []
    with open(source) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict) and "tokens" in obj:
                tokens = [int(t) for t in obj["tokens"]]
                
                # Validate token IDs against vocabulary size
                if vocab_size is not None:
                    max_token = max(tokens) if tokens else 0
                    if max_token >= vocab_size:
                        raise ValueError(
                            f"Fixture {obj.get('category', 'unknown')} contains token ID {max_token} "
                            f"which exceeds vocabulary size {vocab_size}"
                        )
                
                normalized.append(
                    {
                        "category": obj.get("category", "default"),
                        "tokens": tokens,
                        "text": tokenizer.decode(tokens)
                        if hasattr(tokenizer, "decode")
                        else "",
                    }
                )
            elif isinstance(obj, dict):
                text = obj.get("prompt", obj.get("text", ""))
                tokens = tokenizer.encode(text)
                
                # Validate token IDs against vocabulary size
                if vocab_size is not None:
                    max_token = max(tokens) if tokens else 0
                    if max_token >= vocab_size:
                        raise ValueError(
                            f"Encoded text contains token ID {max_token} "
                            f"which exceeds vocabulary size {vocab_size}"
                        )
                
                normalized.append(
                    {
                        "category": "default",
                        "tokens": tokens,
                        "text": text,
                    }
                )
            elif isinstance(obj, str):
                tokens = tokenizer.encode(obj)
                
                # Validate token IDs against vocabulary size
                if vocab_size is not None:
                    max_token = max(tokens) if tokens else 0
                    if max_token >= vocab_size:
                        raise ValueError(
                            f"Encoded text contains token ID {max_token} "
                            f"which exceeds vocabulary size {vocab_size}"
                        )
                
                normalized.append(
                    {
                        "category": "default",
                        "tokens": tokens,
                        "text": obj,
                    }
                )
    return normalized
