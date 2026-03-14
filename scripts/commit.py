#!/usr/bin/env python3
"""
commit.py - Interactive commit message generator with conventional commits support.

Features:
- Interactive prompts for conventional commit types
- Auto-suggest based on changed files
- Scope suggestions
- Body and footer support (BREAKING CHANGE, Closes #issue)
- Preview before committing
- Direct commit option

Usage:
    python commit.py              # Interactive mode
    python commit.py --type feat  # Specify type directly
    python commit.py --no-commit  # Generate message only (don't commit)
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Conventional commit types with descriptions
COMMIT_TYPES = {
    "feat": {"desc": "A new feature", "version": "minor"},
    "fix": {"desc": "A bug fix", "version": "patch"},
    "docs": {"desc": "Documentation only changes", "version": "patch"},
    "style": {"desc": "Changes that don't affect meaning (formatting, etc)", "version": "patch"},
    "refactor": {"desc": "Code change that neither fixes a bug nor adds a feature", "version": "patch"},
    "perf": {"desc": "Performance improvement", "version": "patch"},
    "test": {"desc": "Adding or correcting tests", "version": "patch"},
    "chore": {"desc": "Changes to build process, tooling, or auxiliary files", "version": "patch"},
    "ci": {"desc": "CI configuration changes", "version": "patch"},
}


def run_cmd(cmd, capture=True, check=False):
    """Run shell command and return output."""
    result = subprocess.run(cmd, shell=True, capture_output=capture, text=True, check=check)
    return result.stdout.strip() if capture else result.returncode


def get_staged_files():
    """Get list of staged files."""
    output = run_cmd("git diff --cached --name-only")
    return [f for f in output.split("\n") if f]


def get_unstaged_files():
    """Get list of unstaged files."""
    output = run_cmd("git diff --name-only")
    return [f for f in output.split("\n") if f]


def get_status():
    """Get git status."""
    return run_cmd("git status --short")


def guess_commit_type(files):
    """Guess commit type based on changed files."""
    if not files:
        return "feat"

    docs_keywords = [".md", "docs/", "readme", "changelog", "license"]
    test_keywords = ["test", "spec", "tests/"]
    config_keywords = [".yml", ".yaml", ".toml", ".ini", ".github/", ".gitignore"]
    code_keywords = [".py", ".js", ".ts", ".jsx", ".tsx", ".rs", ".go", ".java"]

    scores = {t: 0 for t in COMMIT_TYPES}

    for f in files:
        f_lower = f.lower()

        if any(k in f_lower for k in docs_keywords):
            scores["docs"] += 2
        if any(k in f_lower for k in test_keywords):
            scores["test"] += 2
        if any(k in f_lower for k in config_keywords):
            scores["ci"] += 1
        if any(k in f_lower for k in code_keywords):
            scores["feat"] += 1

        # Specific patterns
        if "fix" in f_lower or "bug" in f_lower or "issue" in f_lower:
            scores["fix"] += 3
        if "perf" in f_lower or "optim" in f_lower or "speed" in f_lower:
            scores["perf"] += 3
        if "refactor" in f_lower or "clean" in f_lower:
            scores["refactor"] += 3
        if "style" in f_lower or "format" in f_lower or "lint" in f_lower:
            scores["style"] += 3
        if "chore" in f_lower or "update" in f_lower or "upgrade" in f_lower:
            scores["chore"] += 1

    # Return highest scoring type
    best_type = max(scores, key=scores.get)
    if scores[best_type] > 0:
        return best_type

    return "feat"


def guess_scope(files):
    """Guess scope based on common directory structure."""
    if not files:
        return ""

    # Common scopes
    scope_counts = {}
    for f in files:
        parts = f.split("/")
        if len(parts) > 1:
            scope = parts[0]
            scope_counts[scope] = scope_counts.get(scope, 0) + 1

    if scope_counts:
        best_scope = max(scope_counts, key=scope_counts.get)
        if scope_counts[best_scope] >= 2:
            return best_scope

    return ""


def generate_suggestion(commit_type, files, scope=""):
    """Generate a suggested commit message."""
    if not files:
        return f"{commit_type}: update project"

    # Extract meaningful description from files
    if len(files) == 1:
        file_path = files[0]
        file_name = Path(file_path).stem

        if commit_type == "fix":
            return f"{commit_type}: resolve issue in {file_name}"
        elif commit_type == "feat":
            return f"{commit_type}: add {file_name} functionality"
        elif commit_type == "docs":
            return f"{commit_type}: update {file_name} documentation"
        elif commit_type == "test":
            return f"{commit_type}: add tests for {file_name}"
        else:
            return f"{commit_type}: update {file_name}"

    # Multiple files - group by directory or type
    dirs = set()
    for f in files:
        if "/" in f:
            dirs.add(f.split("/")[0])
        else:
            dirs.add(Path(f).stem)

    if len(dirs) == 1:
        target = list(dirs)[0]
        return f"{commit_type}: update {target}"
    elif len(dirs) <= 3:
        return f"{commit_type}: update {', '.join(sorted(dirs))}"
    else:
        return f"{commit_type}: update {len(files)} files"


def display_commit_types():
    """Display available commit types."""
    print("\n📋 Commit Types:")
    print("-" * 60)
    for i, (ctype, info) in enumerate(COMMIT_TYPES.items(), 1):
        print(f"  {i:2}. {ctype:12} - {info['desc']}")
    print("-" * 60)


def display_files(files, staged=True):
    """Display file list."""
    if not files:
        print("  (no files)")
        return

    status_type = "staged" if staged else "unstaged"
    print(f"\n📁 {status_type.title()} files ({len(files)}):")
    for f in files[:20]:
        print(f"     • {f}")
    if len(files) > 20:
        print(f"     ... and {len(files) - 20} more")


def select_commit_type_interactive():
    """Interactive commit type selection."""
    display_commit_types()

    while True:
        try:
            choice = input("\nSelect type (number or name): ").strip().lower()

            # Try numeric choice
            if choice.isdigit():
                idx = int(choice) - 1
                types_list = list(COMMIT_TYPES.keys())
                if 0 <= idx < len(types_list):
                    return types_list[idx]
                print("❌ Invalid number. Try again.")
                continue

            # Try name
            if choice in COMMIT_TYPES:
                return choice

            print(f"❌ '{choice}' is not a valid type. Try again.")
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Cancelled")
            sys.exit(0)


def build_commit_message(ctype, scope, subject, body="", breaking=False, issues=None):
    """Build the complete commit message."""
    # Header
    if scope:
        header = f"{ctype}({scope}): {subject}"
    else:
        header = f"{ctype}: {subject}"

    # Add BREAKING CHANGE indicator
    if breaking:
        if scope:
            header = f"{ctype}({scope})!: {subject}"
        else:
            header = f"{ctype}!: {subject}"

    message = header

    # Add body
    if body:
        message += f"\n\n{body}"

    # Add BREAKING CHANGE footer
    if breaking:
        message += "\n\nBREAKING CHANGE: This change requires updates to existing code."

    # Add issue references
    if issues:
        issue_refs = " ".join(issues)
        message += f"\n\n{issue_refs}"

    return message


def main():
    parser = argparse.ArgumentParser(
        description="Interactive commit message generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python commit.py                    # Full interactive mode
  python commit.py --type feat        # Specify type, interactive rest
  python commit.py --no-commit        # Generate message only
  python commit.py --amend            # Amend last commit
        """,
    )

    parser.add_argument("-t", "--type", choices=list(COMMIT_TYPES.keys()), help="Commit type (feat, fix, docs, etc.)")
    parser.add_argument("-s", "--scope", help="Commit scope (optional)")
    parser.add_argument("-m", "--message", help="Subject line (skip prompts)")
    parser.add_argument("-b", "--body", help="Commit body (optional)")
    parser.add_argument("--breaking", action="store_true", help="Mark as breaking change")
    parser.add_argument("--issue", action="append", help="Issue reference (e.g., 'Closes #123')")
    parser.add_argument("--no-commit", action="store_true", help="Generate message only, don't commit")
    parser.add_argument("--amend", action="store_true", help="Amend last commit")
    parser.add_argument("--all", "-a", action="store_true", help="Stage all changes before committing")

    args = parser.parse_args()

    print("=" * 60)
    print("  📝 Commit Message Generator")
    print("=" * 60)

    # Check git status
    status = get_status()
    if not status and not args.all:
        print("\n❌ No changes to commit.")
        print("   Use 'git add <files>' or run with --all to stage all")
        sys.exit(1)

    # Stage all if requested
    if args.all:
        print("\n📦 Staging all changes...")
        run_cmd("git add -A")
        print("   ✓ Staged")

    # Get staged files
    staged_files = get_staged_files()
    unstaged_files = get_unstaged_files()

    # Show files
    display_files(staged_files, staged=True)
    if unstaged_files:
        display_files(unstaged_files, staged=False)

    # Get commit type
    if args.type:
        commit_type = args.type
    else:
        commit_type = select_commit_type_interactive()

    # Get scope
    if args.scope:
        scope = args.scope
    else:
        suggested_scope = guess_scope(staged_files)
        if suggested_scope:
            scope_input = input(f"\nScope (suggested: '{suggested_scope}'): ").strip()
            scope = scope_input if scope_input else suggested_scope
        else:
            scope = input("\nScope (optional, press Enter to skip): ").strip()

    # Get subject
    if args.message:
        subject = args.message
    else:
        suggestion = generate_suggestion(commit_type, staged_files, scope)
        subject_input = input(f"\nSubject (suggested: '{suggestion}'): ").strip()
        subject = subject_input if subject_input else suggestion

    # Get body
    if args.body:
        body = args.body
    else:
        body = input("\nBody (optional, press Enter to skip): ").strip()

    # Breaking change
    if not args.breaking:
        breaking_input = input("\nBreaking change? (y/N): ").strip().lower()
        breaking = breaking_input in ["y", "yes"]
    else:
        breaking = True

    # Issue references
    issues = args.issue or []
    if not issues:
        issue_input = input("\nIssue references (e.g., 'Closes #123', press Enter to skip): ").strip()
        if issue_input:
            issues = issue_input.split()

    # Build commit message
    commit_msg = build_commit_message(commit_type, scope, subject, body, breaking, issues)

    # Display preview
    print("\n" + "=" * 60)
    print("  📋 Commit Message Preview")
    print("=" * 60)
    print(f"\n{commit_msg}\n")

    # Confirm and commit
    if args.no_commit:
        print("\n💡 To use this message:")
        print(f'   git commit -m "{commit_msg.replace(chr(10), chr(92) + chr(110))}"')
        sys.exit(0)

    confirm = input("\nCommit with this message? (Y/n): ").strip().lower()
    if confirm in ["n", "no"]:
        print("\n👋 Cancelled")
        sys.exit(0)

    # Commit
    print("\n📦 Committing...")
    amend_flag = "--amend" if args.amend else ""
    result = subprocess.run(["git", "commit", amend_flag, "-m", commit_msg], capture_output=True, text=True)

    if result.returncode != 0:
        print("\n❌ Commit failed:")
        print(result.stderr)
        sys.exit(1)

    print("   ✓ Commit successful!")

    # Show commit
    result = run_cmd("git log -1 --oneline")
    print(f"\n📌 Latest commit: {result}")


if __name__ == "__main__":
    main()
