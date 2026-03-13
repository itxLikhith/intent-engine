#!/usr/bin/env python3
"""
commit-gen.py - Generate professional commit messages automatically.

Usage:
    python commit-gen.py              # Auto-generate based on staged changes
    python commit-gen.py --fix        # Generate a fix commit message
    python commit-gen.py --feat       # Generate a feat commit message
    python commit-gen.py --interactive  # Interactive mode
"""

import subprocess
import sys
import argparse

def run_cmd(cmd):
    """Run shell command and return output."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout.strip()

def get_staged_files():
    """Get list of staged files."""
    output = run_cmd("git diff --cached --name-only")
    return output.split('\n') if output else []

def get_diff_stats():
    """Get diff statistics."""
    return run_cmd("git diff --cached --stat")

def guess_commit_type(files):
    """Guess commit type based on changed files."""
    if not files:
        return "feat"
    
    docs_count = sum(1 for f in files if any(x in f.lower() for x in ['.md', 'docs/', 'readme']))
    test_count = sum(1 for f in files if any(x in f.lower() for x in ['test', 'spec']))
    config_count = sum(1 for f in files if any(x in f.lower() for x in ['.yml', '.yaml', '.toml', '.ini', '.github/']))
    
    if docs_count > len(files) / 2:
        return "docs"
    elif test_count > 0 and all('test' in f.lower() for f in files):
        return "test"
    elif config_count > len(files) / 2:
        return "ci"
    elif any('hook' in f or 'lint' in f or 'format' in f or '.pre-commit' in f for f in files):
        return "style"
    
    return "feat"

def generate_subject(commit_type, files):
    """Generate a commit subject line."""
    if not files:
        return f"{commit_type}: update project"
    
    # Group files by directory
    dirs = set()
    for f in files:
        if '/' in f:
            dirs.add(f.split('/')[0])
        else:
            dirs.add(f.split('.')[0] if '.' in f else f)
    
    if len(dirs) == 1:
        target = list(dirs)[0]
        if commit_type == "fix":
            return f"{commit_type}: resolve issues in {target}"
        elif commit_type == "docs":
            return f"{commit_type}: update {target} documentation"
        elif commit_type == "test":
            return f"{commit_type}: add tests for {target}"
        else:
            return f"{commit_type}: improve {target}"
    elif len(dirs) <= 3:
        return f"{commit_type}: update {', '.join(sorted(dirs))}"
    else:
        return f"{commit_type}: update {len(files)} files"

def interactive_mode():
    """Interactive commit message generation."""
    files = get_staged_files()
    
    if not files:
        print("❌ No staged changes. Add files with 'git add' first.")
        sys.exit(1)
    
    print("📝 Staged files:")
    for f in files[:10]:
        print(f"   - {f}")
    if len(files) > 10:
        print(f"   ...and {len(files) - 10} more")
    print()
    
    print("Select commit type:")
    types = ["feat", "fix", "docs", "style", "refactor", "perf", "test", "chore", "ci"]
    for i, t in enumerate(types, 1):
        print(f"   {i}. {t}")
    
    try:
        choice = int(input("\nEnter choice (1-9): ")) - 1
        commit_type = types[choice]
    except (ValueError, IndexError):
        commit_type = guess_commit_type(files)
        print(f"Using default: {commit_type}")
    
    scope = input("Enter scope (optional, press Enter to skip): ").strip()
    subject = input("Enter subject: ").strip()
    
    if not subject:
        subject = generate_subject(commit_type, files)
    
    body = input("Enter body (optional, press Enter to skip): ").strip()
    
    # Build commit message
    if scope:
        commit_msg = f"{commit_type}({scope}): {subject}"
    else:
        commit_msg = f"{commit_type}: {subject}"
    
    if body:
        commit_msg += f"\n\n{body}"
    
    print(f"\n✅ Generated commit message:\n---\n{commit_msg}\n---")
    
    # Option to commit directly
    commit_now = input("\nCommit now? (y/n): ").strip().lower()
    if commit_now == 'y':
        subprocess.run(f'git commit -m "{commit_msg}"', shell=True)
    else:
        print("\n💡 To use this message, run:")
        print(f'   git commit -m "{commit_msg}"')

def main():
    parser = argparse.ArgumentParser(description="Generate professional commit messages")
    parser.add_argument("--fix", action="store_true", help="Generate a fix commit")
    parser.add_argument("--feat", action="store_true", help="Generate a feat commit")
    parser.add_argument("--docs", action="store_true", help="Generate a docs commit")
    parser.add_argument("--test", action="store_true", help="Generate a test commit")
    parser.add_argument("--chore", action="store_true", help="Generate a chore commit")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()
    
    files = get_staged_files()
    
    if not files:
        print("❌ No staged changes. Add files with 'git add' first.")
        sys.exit(1)
    
    if args.interactive:
        interactive_mode()
        return
    
    # Determine commit type
    if args.fix:
        commit_type = "fix"
    elif args.feat:
        commit_type = "feat"
    elif args.docs:
        commit_type = "docs"
    elif args.test:
        commit_type = "test"
    elif args.chore:
        commit_type = "chore"
    else:
        commit_type = guess_commit_type(files)
    
    subject = generate_subject(commit_type, files)
    
    print(f"💡 Suggested commit message:")
    print(f"   git commit -m \"{subject}\"")
    print()
    print("Full diff:")
    print(get_diff_stats())

if __name__ == "__main__":
    main()
