# Commit Message Automation Guide

This project uses automated tools to ensure all commit messages follow the [Conventional Commits](https://www.conventionalcommits.org/) standard.

## Quick Start

```bash
# Stage your changes
git add .

# Option 1: Auto-generate commit message
python commit-gen.py

# Option 2: Interactive mode
python commit-gen.py --interactive

# Option 3: Use git alias
git comg  # Auto-generate and commit
```

## Automated Features

### 1. Commit Message Validation
The `commit-msg` hook automatically validates your commit messages. If the format is incorrect, the commit will be rejected with helpful feedback.

### 2. Auto-Suggestions
The `prepare-commit-msg` hook automatically suggests a commit message based on your staged changes.

### 3. Commit Generator
Run `python commit-gen.py` to generate a professional commit message based on your changes.

## Commit Types

| Type | Description |
|------|-------------|
| `feat` | A new feature |
| `fix` | A bug fix |
| `docs` | Documentation only changes |
| `style` | Formatting, missing semicolons, etc (no code change) |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `perf` | Code change that improves performance |
| `test` | Adding missing tests or correcting existing tests |
| `chore` | Changes to the build process, tooling, or auxiliary files |
| `ci` | Changes to CI configuration files and scripts |

## Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Rules:**
- Subject line max 50 characters
- Body lines max 72 characters
- Use imperative mood in subject ("add" not "added")
- No period at the end of subject line

## Examples

```
feat: add user authentication system

Implemented JWT-based authentication with refresh tokens.
Added rate limiting to prevent brute force attacks.

Closes #123
```

```
fix: resolve database connection timeout

Increased connection pool size from 5 to 20.
Added retry logic with exponential backoff.
```

```
docs: update API documentation

Added examples for all endpoint responses.
Clarified authentication requirements.
```

## Git Aliases

After running `python install_hooks.py`, these aliases are available:

| Alias | Command | Description |
|-------|---------|-------------|
| `git com "msg"` | `git commit -m "msg"` | Quick commit |
| `git comg` | `python commit-gen.py` | Generate commit message |
| `git comi` | `python commit-gen.py --interactive` | Interactive commit |
| `git last` | `git log -1` | Show last commit |
| `git amend` | `git commit --amend` | Amend last commit |
| `git undo` | `git reset --soft HEAD~1` | Undo last commit |

## Installation

Run once to set up all hooks and tools:

```bash
python install_hooks.py
```

This will:
1. Configure the commit message template
2. Install git hooks for validation
3. Create convenient git aliases
4. Set up pre-commit hooks

## Optional: Install Commitizen

For additional commit tools:

```bash
pip install commitizen
```

Then you can use:
- `cz commit` - Interactive commit wizard
- `cz changelog` - Generate changelog
- `cz bump` - Bump version number

## Troubleshooting

### Hook not running?
Make sure hooks are in `.git/hooks/` and executable (Unix: `chmod +x .git/hooks/*`)

### Want to bypass validation?
Use `git commit --no-verify` (use sparingly!)

### Need to fix last commit message?
```bash
git commit --amend -m "new message"
```
