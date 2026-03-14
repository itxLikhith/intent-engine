# GitHub Release Automation

## Overview

The Intent Engine repository is configured for **automatic versioning and release creation** when changes are merged to the `master` branch.

## How It Works

### 1. CI/CD Pipeline Triggers Auto-Version

When the **CI/CD Pipeline** completes successfully on `master`:
```yaml
on:
  workflow_run:
    workflows: ["CI/CD Pipeline"]
    types:
      - completed
    branches: [main, master]
```

### 2. Auto-Version Workflow Runs

The **Auto Version Bump** workflow:
1. Analyzes commits since last tag
2. Determines version bump type (patch/minor/major)
3. Syncs version with last tag if mismatch detected
4. Skips if no new commits
5. Bumps version using Commitizen
6. Creates git tag
7. Pushes changes to GitHub
8. **Creates GitHub Release automatically**

### 3. Release Creation

The workflow automatically creates a GitHub release with:
- **Tag:** `v{version}` (e.g., `v0.3.0`)
- **Title:** `Release {version}`
- **Body:** Auto-generated from commits
- **Release Notes:** Generated from commit messages

## Version Bump Rules

| Commit Type | Bump Type | Example |
|-------------|-----------|---------|
| `fix:` | PATCH | `v0.2.3` → `v0.2.4` |
| `perf:` | PATCH | `v0.2.3` → `v0.2.4` |
| `feat:` | MINOR | `v0.2.3` → `v0.3.0` |
| `BREAKING CHANGE` | MAJOR | `v0.2.3` → `v1.0.0` |
| `feat!:` | MAJOR | `v0.2.3` → `v1.0.0` |

## Error Handling

The workflow includes comprehensive error handling:

### ✅ Version Mismatch
- Detects if `pyproject.toml` version doesn't match last tag
- Automatically syncs version before bumping

### ✅ No New Commits
- Checks for commits since last tag
- Skips version bump if no changes

### ✅ Tag Already Exists
- Checks if tag exists before creating
- Gracefully skips if already tagged
- Logs warning instead of failing

### ✅ Release Creation
- Uses `softprops/action-gh-release@v1`
- Continues on error (doesn't break pipeline)
- Generates release notes automatically

## Manual Trigger

You can manually trigger the auto-version workflow:

```bash
# Via GitHub Actions UI
1. Go to Actions → Auto Version Bump
2. Click "Run workflow"
3. Select branch (master)
4. Click "Run workflow"
```

## Required Secrets

The workflow requires these GitHub secrets:
- `GITHUB_TOKEN` (automatically provided)

## Example Flow

### Scenario: Fix Commit Merged

```bash
# Developer merges PR with:
git commit -m "fix: update script paths in CI"

# Automated flow:
1. CI/CD Pipeline runs → ✅ Success
2. Auto Version workflow triggers
3. Analyzes commits → detects "fix:" → PATCH bump
4. Bumps version: v0.3.0 → v0.3.1
5. Creates tag: v0.3.0
6. Pushes to master
7. Creates GitHub Release v0.3.1
```

## Current Configuration

**Workflow File:** `.github/workflows/auto-version.yml`

**Key Features:**
- ✅ Automatic version bumping
- ✅ Automatic tag creation
- ✅ Automatic GitHub release
- ✅ Version sync with last tag
- ✅ Skip if no new commits
- ✅ Tag existence checks
- ✅ Graceful error handling
- ✅ Generated release notes

## Troubleshooting

### Release Not Created?

Check:
1. CI/CD Pipeline passed successfully
2. Workflow has `contents: write` permission
3. No existing tag with same version
4. Check workflow logs in Actions tab

### Version Not Bumping?

Check:
1. Commits follow Conventional Commits format
2. Commits since last tag exist
3. Version in `pyproject.toml` matches last tag
4. Check workflow logs for bump type determination

### Tag Already Exists Error?

This is handled gracefully:
- Workflow detects existing tag
- Skips tag creation
- Logs warning
- Continues without failure

## Next Release

**Next version:** `v0.3.1` (assuming next commit is a `fix:` or `perf:`)

**To trigger:**
1. Merge any commit to `master`
2. Wait for CI/CD to complete
3. Auto-version workflow runs automatically
4. Release created automatically!

---

**Status:** ✅ **Auto-release enabled and configured**
