#!/usr/bin/env python3
"""
Setup script for Git hooks and commit automation.
Run this once to configure professional commit messages.
"""

import subprocess
import sys
import os

def run_cmd(cmd, check=False):
    """Run shell command."""
    print(f"  → {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"     ⚠️  Warning: {result.stderr.strip()}")
    return result

def main():
    print("=" * 60)
    print("  Git Hooks & Commit Automation Setup")
    print("=" * 60)
    print()
    
    # Check if we're in a git repo
    result = run_cmd("git rev-parse --git-dir")
    if result.returncode != 0:
        print("❌ Not a git repository. Please run this from the project root.")
        sys.exit(1)
    
    print("✓ Git repository detected\n")
    
    # 1. Set commit template
    print("1. Setting up commit message template...")
    run_cmd("git config commit.template .gitmessage")
    print("   ✓ Commit template configured\n")
    
    # 2. Make hooks executable (Windows doesn't need this)
    if os.name != 'nt':
        print("2. Making hooks executable...")
        run_cmd("chmod +x .git/hooks/commit-msg", check=True)
        run_cmd("chmod +x .git/hooks/prepare-commit-msg", check=True)
        print("   ✓ Hooks made executable\n")
    else:
        print("2. Skipping chmod (Windows)\n")
    
    # 3. Create git aliases
    print("3. Creating git aliases...")
    aliases = {
        'com': 'commit -m',
        'comg': '!python commit-gen.py',
        'comi': '!python commit-gen.py --interactive',
        'last': 'log -1 --pretty=format:"%h %s"',
        'amend': 'commit --amend',
        'undo': 'reset --soft HEAD~1',
    }
    
    for alias, cmd in aliases.items():
        run_cmd(f'git config --global alias.{alias} "{cmd}"', check=True)
    print("   ✓ Git aliases created\n")
    
    # 4. Install pre-commit hooks
    print("4. Installing pre-commit hooks...")
    result = run_cmd("pre-commit install --hook-type commit-msg", check=True)
    if result.returncode == 0:
        print("   ✓ Pre-commit hooks installed\n")
    else:
        print("   ⚠️  Pre-commit not installed. Run: pip install pre-commit\n")
    
    # 5. Install commitizen (optional)
    print("5. Checking commitizen...")
    result = run_cmd("python -m commitizen --version", check=True)
    if result.returncode == 0:
        print("   ✓ Commitizen is installed\n")
    else:
        print("   ⚠️  Commitizen not installed. Run: pip install commitizen\n")
    
    # Summary
    print("=" * 60)
    print("  Setup Complete!")
    print("=" * 60)
    print()
    print("Available commands:")
    print("  python commit-gen.py           - Auto-generate commit message")
    print("  python commit-gen.py -i        - Interactive commit generation")
    print("  git com \"message\"              - Quick commit")
    print("  git comg                       - Generate and commit")
    print("  git comi                       - Interactive commit")
    print()
    print("Commit types:")
    print("  feat:     New feature")
    print("  fix:      Bug fix")
    print("  docs:     Documentation")
    print("  style:    Formatting")
    print("  refactor: Code restructuring")
    print("  perf:     Performance improvement")
    print("  test:     Tests")
    print("  chore:    Maintenance")
    print("  ci:       CI/CD changes")
    print()

if __name__ == "__main__":
    main()
