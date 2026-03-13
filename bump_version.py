#!/usr/bin/env python3
"""
Version bump script for intent-engine.

This script provides manual control over version bumping,
complementing the automatic Commitizen versioning.

Usage:
    python bump_version.py [--major | --minor | --patch] [--dry-run]

Examples:
    python bump_version.py --patch      # Bump patch version: 0.1.0 -> 0.1.1
    python bump_version.py --minor      # Bump minor version: 0.1.0 -> 0.2.0
    python bump_version.py --major      # Bump major version: 0.1.0 -> 1.0.0
    python bump_version.py --dry-run    # Show what would change without applying
"""

import argparse
import re
import sys
from pathlib import Path


def get_current_version(version_file: Path) -> str:
    """Extract current version from __version__.py file."""
    content = version_file.read_text()
    match = re.search(r'__version__\s+=\s+"([^"]+)"', content)
    if not match:
        raise ValueError(f"Could not find __version__ in {version_file}")
    return match.group(1)


def bump_version(version: str, bump_type: str) -> str:
    """Bump version based on semantic versioning."""
    major, minor, patch = map(int, version.split("."))

    if bump_type == "major":
        major += 1
        minor = 0
        patch = 0
    elif bump_type == "minor":
        minor += 1
        patch = 0
    elif bump_type == "patch":
        patch += 1
    else:
        raise ValueError(f"Invalid bump type: {bump_type}")

    return f"{major}.{minor}.{patch}"


def update_version_file(version_file: Path, new_version: str) -> None:
    """Update version in __version__.py file."""
    content = version_file.read_text()
    new_content = re.sub(
        r'(__version__\s+=\s+)"[^"]+"',
        rf'\1"{new_version}"',
        content,
    )
    version_file.write_text(new_content)


def update_pyproject_toml(pyproject_file: Path, new_version: str) -> None:
    """Update version in pyproject.toml file."""
    content = pyproject_file.read_text()
    new_content = re.sub(
        r'(^version\s+=\s+)"[^"]+"',
        rf'\1"{new_version}"',
        content,
        flags=re.MULTILINE,
    )
    pyproject_file.write_text(new_content)


def main():
    parser = argparse.ArgumentParser(
        description="Bump version for intent-engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--major",
        action="store_true",
        help="Bump major version (X.0.0)",
    )
    parser.add_argument(
        "--minor",
        action="store_true",
        help="Bump minor version (x.Y.0)",
    )
    parser.add_argument(
        "--patch",
        action="store_true",
        help="Bump patch version (x.y.Z)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without applying",
    )

    args = parser.parse_args()

    # Determine bump type
    bump_types = sum([args.major, args.minor, args.patch])
    if bump_types == 0:
        parser.error("Must specify one of: --major, --minor, --patch")
    if bump_types > 1:
        parser.error("Can only specify one of: --major, --minor, --patch")

    bump_type = "major" if args.major else "minor" if args.minor else "patch"

    # File paths
    root_dir = Path(__file__).parent
    version_file = root_dir / "__version__.py"
    pyproject_file = root_dir / "pyproject.toml"

    # Get current version
    try:
        current_version = get_current_version(version_file)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Calculate new version
    new_version = bump_version(current_version, bump_type)

    # Display results
    print(f"Current version: {current_version}")
    print(f"Bump type: {bump_type}")
    print(f"New version: {new_version}")

    if args.dry_run:
        print("\n[DRY RUN] No files were modified.")
        print("Run without --dry-run to apply changes.")
        return

    # Apply changes
    update_version_file(version_file, new_version)
    update_pyproject_toml(pyproject_file, new_version)

    print(f"\n✓ Version bumped to {new_version}")
    print("Files updated:")
    print(f"  - {version_file}")
    print(f"  - {pyproject_file}")
    print("\nNext steps:")
    print("  1. Commit the changes: git add __version__.py pyproject.toml")
    print(f"  2. Create a commit: git commit -m 'chore: bump version to {new_version}'")
    print(f"  3. Create a tag: git tag {new_version}")
    print("  4. Push: git push && git push --tags")


if __name__ == "__main__":
    main()
