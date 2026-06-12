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
    try:
        from packaging.version import Version
    except ImportError:
        raise RuntimeError("packaging is required for version parsing; install it with: pip install packaging")
    v = Version(current)
    major, minor, micro = v.major, v.minor, v.micro
    if part == "major":
        major += 1
        minor = 0
        micro = 0
    elif part == "minor":
        minor += 1
        micro = 0
    elif part == "patch":
        micro += 1
    else:
        raise ValueError(f"Unknown bump part: {part}")
    return f"{major}.{minor}.{micro}"


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

    # 1. Start from a clean tree.
    if not _git_tree_clean():
        print("ERROR: git tree is dirty. Commit or stash changes before releasing.")
        sys.exit(1)

    # 2. Update version.
    if args.bump:
        new_version = _bump_version(current, args.bump)
        print(f"Bumping {args.bump}: {current} -> {new_version}")
        _set_version(new_version)
        print("Version updated. Commit the change before proceeding.")
        sys.exit(0)
    else:
        new_version = current

    # 3. Build.
    if not args.skip_build:
        print("Building sdist and wheel...")
        _run([sys.executable, "-m", "pip", "install", "build"])
        _run([sys.executable, "-m", "build", "--sdist", "--wheel"])

    # 4. Run tests.
    if not args.skip_tests:
        print("Running test suite...")
        _run([sys.executable, "-m", "pytest", "tests/", "-q"])

    # 5. Create signed tag.
    if not args.skip_tag:
        tag = f"v{new_version}"
        print(f"Creating git tag {tag}...")
        _run(["git", "tag", "-a", tag, "-m", f"Release {tag}"])

    print("\nRelease prepared locally.")
    print("Review the build artifacts, then push the tag:")
    print(f"  git push origin main && git push origin v{new_version}")


if __name__ == "__main__":
    main()
