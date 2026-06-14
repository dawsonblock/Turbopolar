"""Deterministic, exact-length prompt fixtures for real-model benchmarks.

The fixtures in ``exact_token_fixtures.jsonl`` store explicit token-id sequences
so that prompt lengths are independent of tokenizer merging behavior.  This makes
perplexity, compression, and decode-speed comparisons reproducible across models.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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
) -> List[Dict[str, Any]]:
    """Build a deterministic prompt fixture list with exact token lengths.

    Args:
        specs: list of (category, length) pairs.  Defaults to a short-to-stress
            progression aligned with the 64-token TurboPolar block size.
        start_id: smallest token id to use; must be < model vocab size.
        vocab_stride: number of distinct token ids to cycle through.

    Returns:
        List of fixture dicts with ``category``, ``length``, and ``tokens`` keys.
    """
    specs = specs or CATEGORY_LENGTHS
    fixtures = []
    for category, length in specs:
        tokens = _token_sequence(length, start_id=start_id, vocab_stride=vocab_stride)
        fixtures.append({"category": category, "length": length, "tokens": tokens})
    return fixtures


def write_token_fixtures(
    fixtures: List[Dict[str, Any]],
    path: Path,
) -> Path:
    """Write fixture dicts to a JSONL file, one fixture per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for fixture in fixtures:
            f.write(json.dumps(fixture) + "\n")
    return path


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
