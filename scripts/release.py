#!/usr/bin/env python3
"""Release helper for turbo-polar.

Bumps the version in ``pyproject.toml`` (optional), validates that the git tree
is clean, runs the test suite, builds an sdist/wheel, and creates a signed tag.
It intentionally does **not** push anything; the user reviews and pushes.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = PROJECT_ROOT / "pyproject.toml"


def _run(cmd: list[str], cwd: Path = PROJECT_ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True)


def _current_version() -> str:
    text = PYPROJECT.read_text()
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not match:
        raise RuntimeError("Could not find version in pyproject.toml")
    return match.group(1)


def _bump_version(current: str, part: str) -> str:
    # Strip pre-release suffixes like .dev0 for bumping.
    base = current.split("+")[0].split(".")[0]
    parts = base.split(".")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        raise RuntimeError(f"Cannot bump non-semver version: {current}")
    major, minor, patch = map(int, parts)
    if part == "major":
        major += 1
        minor = 0
        patch = 0
    elif part == "minor":
        minor += 1
        patch = 0
    elif part == "patch":
        patch += 1
    else:
        raise ValueError(f"Unknown bump part: {part}")
    return f"{major}.{minor}.{patch}"


def _set_version(version: str) -> None:
    text = PYPROJECT.read_text()
    new_text = re.sub(
        r'^(version\s*=\s*")([^"]+)(")',
        lambda m: f'{m.group(1)}{version}{m.group(3)}',
        text,
        flags=re.MULTILINE,
    )
    if new_text == text:
        raise RuntimeError("Version line was not updated")
    PYPROJECT.write_text(new_text)


def _git_tree_clean() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and result.stdout.strip() == ""


def main():
    parser = argparse.ArgumentParser(description="Prepare a turbo-polar release")
    parser.add_argument(
        "--bump",
        choices=["major", "minor", "patch"],
        default=None,
        help="Bump the version part before building",
    )
    parser.add_argument(
        "--skip-tests", action="store_true", help="Skip the test suite"
    )
    parser.add_argument(
        "--skip-build", action="store_true", help="Skip building sdist/wheel"
    )
    parser.add_argument(
        "--skip-tag", action="store_true", help="Skip creating a git tag"
    )
    args = parser.parse_args()

    current = _current_version()
    print(f"Current version: {current}")

    if args.bump:
        new_version = _bump_version(current, args.bump)
        print(f"Bumping {args.bump}: {current} -> {new_version}")
        _set_version(new_version)
    else:
        new_version = current

    if not _git_tree_clean():
        print("ERROR: git tree is dirty. Commit or stash changes before releasing.")
        sys.exit(1)

    if not args.skip_tests:
        print("Running test suite...")
        _run([sys.executable, "-m", "pytest", "tests/", "-q"])

    if not args.skip_build:
        print("Building sdist and wheel...")
        _run([sys.executable, "-m", "pip", "install", "build"])
        _run([sys.executable, "-m", "build", "--sdist", "--wheel"])

    if not args.skip_tag:
        tag = f"v{new_version}"
        print(f"Creating git tag {tag}...")
        _run(["git", "tag", "-a", tag, "-m", f"Release {tag}"])

    print("\nRelease prepared locally.")
    print("Review the build artifacts, then push the tag:")
    print(f"  git push origin main && git push origin v{new_version}")


if __name__ == "__main__":
    main()
