# Git Hooks & Commit Automation Setup

This script sets up automated professional commit messages for the Intent Engine project.

## Usage

Run this script once to set up all hooks and tools:

```bash
# Windows
python setup_git_hooks.py

# Unix/Mac
python setup_git_hooks.py
```

## What This Sets Up

1. **Git commit template** - Provides a template for commit messages
2. **Commit-msg hook** - Validates commit message format
3. **Prepare-commit-msg hook** - Auto-suggests commit messages
4. **commit-gen.py** - CLI tool for generating commit messages
5. **Git aliases** - Shortcuts for common commit operations

## Git Aliases Created

- `git com` - Quick commit with message
- `git comg` - Generate commit message automatically
- `git comi` - Interactive commit message generation
- `git last` - Show last commit message
- `git amend` - Amend last commit

## Manual Installation (if script fails)

### 1. Set commit template
```bash
git config commit.template .gitmessage
```

### 2. Make hooks executable (Unix/Mac only)
```bash
chmod +x .git/hooks/commit-msg
chmod +x .git/hooks/prepare-commit-msg
```

### 3. Install commitizen (optional)
```bash
pip install pre-commit commitizen
pre-commit install --hook-type commit-msg
```

## Usage Examples

### Auto-generate commit message
```bash
git add .
python commit-gen.py
# Shows suggested commit message
```

### Interactive mode
```bash
git add .
python commit-gen.py --interactive
```

### Force specific commit type
```bash
git add .
python commit-gen.py --fix
python commit-gen.py --feat
python commit-gen.py --docs
```

### Using git aliases
```bash
git add .
git comg  # Auto-generate and commit
git comi  # Interactive commit
```
