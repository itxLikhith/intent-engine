#!/usr/bin/env python3
"""
autopush.py - Automate commit message generation, checks, and push.

Workflow: stage -> check -> commit -> fetch & rebase -> push
Handles CI linting commits (auto-fix [skip ci]) by rebasing on top of them.

Usage:
    python autopush.py                # Stage all, generate msg, check, commit, rebase, push
    python autopush.py --no-check     # Skip lint/format checks
    python autopush.py --dry-run      # Show what would happen without doing it
    python autopush.py -m "message"   # Override auto-generated commit message
    python autopush.py --staged       # Only commit already-staged files

Aliases (Makefile):
    make p                            # Same as: python autopush.py
    make q                            # Same as: python autopush.py --no-check
    make f                            # Same as: python autopush.py --fix
    make d                            # Same as: python autopush.py --dry-run
"""

import argparse
import importlib.util
import io
import os
import subprocess
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

_script_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("commit_gen", os.path.join(_script_dir, "commit-gen.py"))
_commit_gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_commit_gen)

get_staged_files = _commit_gen.get_staged_files
guess_commit_type = _commit_gen.guess_commit_type
generate_subject = _commit_gen.generate_subject
get_diff_stats = _commit_gen.get_diff_stats


def run_cmd(cmd, capture=True):
    """Run a command (list or string) and return (returncode, stdout, stderr)."""
    use_shell = isinstance(cmd, str)
    result = subprocess.run(cmd, shell=use_shell, capture_output=capture, text=True, encoding="utf-8", errors="replace")
    return result.returncode, (result.stdout or "").strip(), (result.stderr or "").strip()


def has_changes():
    """Check if there are any uncommitted changes (staged or unstaged)."""
    code, out, _ = run_cmd("git status --porcelain")
    return bool(out)


def has_staged_changes():
    """Check if there are staged changes."""
    code, out, _ = run_cmd("git diff --cached --name-only")
    return bool(out)


def get_all_changed_files():
    """Get all changed files (staged + unstaged + untracked) via git status."""
    result = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    raw = result.stdout.rstrip()
    if not raw:
        return []
    files = []
    for line in raw.splitlines():
        if not line or len(line) < 4:
            continue
        # Porcelain v1 format: "XY PATH" or "XY ORIG -> PATH"
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ")[1]
        path = path.strip()
        if path:
            files.append(path)
    return files


def stage_all():
    """Stage all changes."""
    code, _, err = run_cmd(["git", "add", "-A"])
    if code != 0:
        print(f"[X] Failed to stage changes: {err}")
        sys.exit(1)
    print("[+] Staged all changes")


def run_checks():
    """Run lint and format checks. Returns True if passed."""
    print("\nRunning checks...")

    print("  > ruff check .")
    code, out, err = run_cmd("ruff check .")
    if code != 0:
        print(f"  [X] Lint failed:\n{out or err}")
        return False
    print("  [+] Lint passed")

    print("  > ruff format --check .")
    code, out, err = run_cmd("ruff format --check .")
    if code != 0:
        print(f"  [X] Format check failed:\n{out or err}")
        print("  Tip: Run 'ruff format .' to auto-fix")
        return False
    print("  [+] Format passed")

    return True


def commit(message):
    """Commit with the given message. Returns True if successful."""
    code, out, err = run_cmd(["git", "commit", "-m", message])
    if code != 0:
        print(f"[X] Commit failed: {err or out}")
        return False
    print(f"[+] Committed: {message}")
    return True


def push():
    """Push to remote. Returns True if successful."""
    print("\nPushing to remote...")
    code, out, err = run_cmd(["git", "push"], capture=False)
    if code != 0:
        print("\n[X] Push failed")
        return False
    print("[+] Pushed successfully")
    return True


def revert_commit():
    """Revert the last commit (soft reset, keeps changes staged)."""
    print("\nReverting commit (soft reset)...")
    code, _, err = run_cmd(["git", "reset", "--soft", "HEAD~1"])
    if code != 0:
        print(f"[X] Failed to revert: {err}")
        print("   Manual fix: git reset --soft HEAD~1")
        sys.exit(1)
    print("[+] Commit reverted. Your changes are still staged.")


def get_current_branch():
    """Get current git branch name."""
    _, branch, _ = run_cmd("git rev-parse --abbrev-ref HEAD")
    return branch


def has_remote():
    """Check if current branch has a remote tracking branch."""
    code, _, _ = run_cmd("git rev-parse --abbrev-ref @{upstream}")
    return code == 0


def sync_with_remote(branch):
    """Fetch and rebase on top of remote changes (e.g. CI linting fixes).

    Returns True if sync succeeded or was not needed, False on rebase conflict.
    """
    print("\nSyncing with remote...")

    # Fetch latest from origin
    code, _, err = run_cmd(["git", "fetch", "origin"])
    if code != 0:
        print(f"  [!] Fetch failed: {err}")
        print("  Continuing anyway (push may fail if remote is ahead)")
        return True

    # Check if remote is ahead
    code, counts, _ = run_cmd(f"git rev-list --left-right --count HEAD...origin/{branch}")
    if code != 0:
        print("  [!] Could not compare with remote, continuing...")
        return True

    parts = counts.split()
    if len(parts) != 2:
        return True

    behind = int(parts[1])
    if behind == 0:
        print("  [+] Already up to date with remote")
        return True

    # Show what remote commits we need to rebase onto
    print(f"  Remote is {behind} commit(s) ahead. Inspecting...")
    _, log_out, _ = run_cmd(f"git log --oneline HEAD..origin/{branch}")
    if log_out:
        ci_lint_marker = "[skip ci]"
        for line in log_out.splitlines():
            is_ci = ci_lint_marker in line or "auto-fix linting" in line.lower()
            tag = " (CI lint fix)" if is_ci else ""
            print(f"    {line}{tag}")

    # Pull with rebase
    print(f"\n  Rebasing onto origin/{branch}...")
    code, out, err = run_cmd(["git", "pull", "--rebase", "--no-verify", "origin", branch])
    if code != 0:
        print(f"  [X] Rebase failed: {err or out}")
        print("  Aborting rebase...")
        run_cmd(["git", "rebase", "--abort"])
        return False

    print("  [+] Rebase successful")
    return True


def main():
    parser = argparse.ArgumentParser(description="Auto commit, check, and push")
    parser.add_argument("-m", "--message", help="Override auto-generated commit message")
    parser.add_argument("--no-check", action="store_true", help="Skip lint/format checks")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen")
    parser.add_argument("--staged", action="store_true", help="Only commit already-staged files")
    parser.add_argument("--fix", action="store_true", help="Auto-fix lint/format before committing")
    args = parser.parse_args()

    print("=" * 50)
    print("  Autopush - Commit & Push Automation")
    print("=" * 50)

    branch = get_current_branch()
    print(f"\nBranch: {branch}")

    # 1. Check for changes
    if not has_changes() and not has_staged_changes():
        print("\n[X] No changes to commit.")
        sys.exit(0)

    # 2. Stage changes (or preview what would be staged)
    if not args.staged:
        if args.dry_run:
            print("\n[dry-run] Would stage all changes")
            files = get_all_changed_files()
        else:
            stage_all()
            files = get_staged_files()
    else:
        if not has_staged_changes():
            print("\n[X] No staged changes. Use 'git add' or drop --staged.")
            sys.exit(1)
        print("\n[+] Using already-staged changes")
        files = get_staged_files()

    # 3. Show files
    print(f"\nFiles ({len(files)}):")
    for f in files[:15]:
        print(f"   - {f}")
    if len(files) > 15:
        print(f"   ...and {len(files) - 15} more")

    # 4. Auto-fix if requested
    if args.fix:
        print("\nAuto-fixing lint/format...")
        if not args.dry_run:
            run_cmd("ruff check . --fix")
            run_cmd("ruff format .")
            run_cmd(["git", "add", "-A"])
            print("  [+] Auto-fix applied")
            files = get_staged_files()
        else:
            print("  [dry-run] Would run ruff check --fix and ruff format")

    # 5. Run checks
    if not args.no_check:
        if args.dry_run:
            print("\n[dry-run] Would run lint/format checks")
        else:
            if not run_checks():
                print("\nOptions:")
                print("   - Fix issues and retry")
                print("   - Run with --fix to auto-fix")
                print("   - Run with --no-check to skip checks")
                sys.exit(1)
    else:
        print("\nSkipping checks (--no-check)")

    # 6. Generate commit message
    if args.message:
        commit_msg = args.message
    else:
        commit_type = guess_commit_type(files)
        commit_msg = generate_subject(commit_type, files)

    print(f"\nCommit message: {commit_msg}")

    # 7. Show diff stats
    stats = get_diff_stats()
    if stats:
        print(f"\nStats:\n{stats}")

    # 8. Dry run stops here
    if args.dry_run:
        print("\n[dry-run] Would commit, rebase, and push. No changes made.")
        sys.exit(0)

    # 9. Commit
    if not commit(commit_msg):
        sys.exit(1)

    # 10. Sync with remote (fetch + rebase over CI linting commits)
    if has_remote():
        if not sync_with_remote(branch):
            print("\n[X] Rebase conflict. Reverting your commit (changes stay staged).")
            revert_commit()
            print("\nTo fix manually:")
            print("   1. git pull --rebase origin " + branch)
            print("   2. Resolve conflicts")
            print("   3. git rebase --continue")
            print("   4. Re-run: python autopush.py")
            sys.exit(1)

    # 11. Push
    if not has_remote():
        print(f"\nNo upstream set. Setting upstream to origin/{branch}...")
        code, _, err = run_cmd(["git", "push", "-u", "origin", branch], capture=False)
        if code != 0:
            print("\n[X] Push failed")
            revert_commit()
            sys.exit(1)
        print("[+] Pushed and set upstream")
    else:
        if not push():
            revert_commit()
            sys.exit(1)

    print("\n" + "=" * 50)
    print("  All done! Committed and pushed.")
    print("  (versioning handled automatically by CI)")
    print("=" * 50)


if __name__ == "__main__":
    main()
