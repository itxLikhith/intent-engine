# Intent Engine - Documentation Organization Guide

**Purpose:** Keep documentation organized and easy to navigate

---

## 📁 Root Directory Files

**Keep these in root (DO NOT MOVE):**

| File | Purpose |
|------|---------|
| `README.md` | Main project README - first thing users see |
| `INDEX.md` | Documentation index - central navigation |
| `CONTRIBUTING.md` | Contribution guidelines |
| `LICENSE` | License file |
| `CHANGELOG_v2.md` | Current version changelog |
| `V2_SUMMARY.md` | v2.0 release summary |
| `QWEN.md` | Project context for AI assistants |

**Why:** These are essential files that users expect in the root directory.

---

## 📁 docs/ Directory Structure

```
docs/
├── README.md                    # Documentation README (optional)
├── ORGANIZATION.md              # This file - organization guide
├── INTEGRATION_GUIDE.md         # v2.0 integration guide
│
├── getting-started/             # Quick start guides
│   ├── QUICKSTART.md
│   ├── README_PRODUCTION.md
│   ├── README_PRODUCTION_FULL.md
│   ├── QUICK_REFERENCE.md
│   └── PHASE1_README.md         # Historical (moved from root)
│
├── deployment/                  # Production deployment
│   ├── DEPLOYMENT_CHECKLIST.md
│   ├── PERFORMANCE_OPTIMIZATION_PLAN.md
│   ├── CI_IMPROVEMENTS.md
│   ├── RELEASE_AUTOMATION.md
│   └── IMPLEMENTATION_GUIDE.md  # Moved from root
│
├── architecture/                # System design
│   ├── PROJECT_OVERVIEW.md
│   ├── PROJECT_STRUCTURE.md
│   ├── Intent-Engine-Whitepaper.md
│   ├── SELF_IMPROVING_LOOP.md   # NEW - v2.0 architecture
│   └── ARCHITECTURE_SUMMARY.md  # Moved from root
│
├── go-crawler/                  # Go crawler docs
│   ├── README.md
│   ├── README_PRODUCTION.md
│   ├── QUICKSTART.md
│   ├── GO_CRAWLER_SETUP_GUIDE.md
│   ├── GO_CRAWLER_INDEXER_PLAN.md
│   ├── INTENT_INDEXER_README.md
│   ├── PHASE_1_IMPLEMENTATION.md
│   ├── IMPLEMENTATION_SUMMARY.md
│   ├── GO_CRAWLER_PRODUCTION_SUMMARY.md
│   ├── FILE_REFERENCE.md
│   ├── BUG_FIXES.md
│   ├── CRAWLER_TEST_RESULTS.md
│   ├── DOCKER_TEST_RESULTS.md
│   ├── RELIABLE_TOPIC_DISCOVERY.md  # Moved from root
│   └── SEED_DISCOVERY_README.md     # Moved from root
│
├── reference/                   # Technical reference
│   ├── Intent-Engine-Tech-Reference.md
│   ├── Intent-Engine-Visual-Guide.md
│   ├── COMPREHENSIVE_GUIDE.md
│   ├── CONFIGURATION_CHANGES.md
│   ├── VERSIONING.md
│   ├── CHANGELOG.md             # Moved from root
│   └── VERSIONING_AND_RELEASES.md
│
└── testing/                     # Testing guides
    └── (test documentation)
```

---

## 📋 Organization Rules

### 1. Root Directory
- **Maximum 10 markdown files** (excluding LICENSE, .md files only)
- **Essential files only** - README, INDEX, CONTRIBUTING, LICENSE, current changelog
- **No historical docs** - Move old docs to appropriate subfolder

### 2. docs/ Subfolders
- **getting-started/** - Quick starts, tutorials, onboarding
- **deployment/** - Production setup, operations, CI/CD
- **architecture/** - System design, diagrams, whitepapers
- **go-crawler/** - Go-specific documentation
- **reference/** - API docs, configuration, changelogs
- **testing/** - Test plans, performance results

### 3. File Naming
- **UPPERCASE** for main documents (README.md, INDEX.md)
- **PascalCase** for guides (QuickStart.md, IntegrationGuide.md)
- **Descriptive names** - Clear what the doc contains

### 4. When Adding New Docs
1. **Determine category** - Which folder fits best?
2. **Check for duplicates** - Does similar doc exist?
3. **Update INDEX.md** - Add to documentation index
4. **Cross-reference** - Link to related docs

### 5. When Moving Docs
1. **Update INDEX.md** - Fix all references
2. **Check for broken links** - Search for old paths
3. **Add redirect note** - Optional: add note in old location
4. **Update changelog** - Document the reorganization

---

## 🔄 Recent Reorganization (v2.0)

**Moved to docs/:**
- `CHANGELOG.md` → `docs/reference/CHANGELOG.md`
- `ARCHITECTURE_SUMMARY.md` → `docs/architecture/ARCHITECTURE_SUMMARY.md`
- `IMPLEMENTATION_GUIDE.md` → `docs/deployment/IMPLEMENTATION_GUIDE.md`
- `PHASE1_README.md` → `docs/getting-started/PHASE1_README.md`
- `RELIABLE_TOPIC_DISCOVERY.md` → `docs/go-crawler/RELIABLE_TOPIC_DISCOVERY.md`
- `SEED_DISCOVERY_README.md` → `docs/go-crawler/SEED_DISCOVERY_README.md`

**Kept in root:**
- `README.md` - Main entry point
- `INDEX.md` - Documentation index
- `CHANGELOG_v2.md` - Current version
- `V2_SUMMARY.md` - Current release summary
- `CONTRIBUTING.md` - Contribution guide
- `LICENSE` - License file
- `QWEN.md` - AI context file

---

## 📊 Documentation Statistics

| Category | Files | Purpose |
|----------|-------|---------|
| Root | 7 | Essential files only |
| getting-started | 5 | Quick starts |
| deployment | 5 | Production ops |
| architecture | 5 | System design |
| go-crawler | 15 | Go crawler docs |
| reference | 7 | Technical reference |
| testing | 3 | Testing guides |
| **Total** | **47** | **Organized!** |

---

## ✅ Maintenance Checklist

**Monthly:**
- [ ] Check for orphaned docs (not linked from INDEX.md)
- [ ] Update version numbers in docs
- [ ] Remove outdated documentation
- [ ] Verify all links work

**Per Release:**
- [ ] Add changelog entry
- [ ] Update version in README and INDEX
- [ ] Add new features to integration guide
- [ ] Update architecture docs if needed

---

## 🎯 Quick Reference

**Looking for:**
- **How to start?** → `docs/getting-started/QUICKSTART.md`
- **Production setup?** → `docs/deployment/`
- **Architecture?** → `docs/architecture/`
- **Go crawler?** → `docs/go-crawler/`
- **API reference?** → `docs/reference/` or `http://localhost:8000/docs`
- **Changelog?** → `CHANGELOG_v2.md` (current) or `docs/reference/CHANGELOG.md` (historical)

---

**Last Updated:** March 15, 2026  
**Maintained By:** Documentation Team
