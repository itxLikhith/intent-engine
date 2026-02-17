# Intent Engine - Parent Directory Structure Guide

This document outlines the recommended structure for the parent directory (`intent-ads/`) and which files should be placed there.

## Recommended Parent Directory Structure

```
intent-ads/
├── README.md                     # Main project README (copy from intent-engine/)
├── LICENSE                       # MIT License file
├── .gitignore                    # Git ignore rules (copy from intent-engine/)
├── docker-compose.yml            # Root Docker Compose (links to intent-engine/)
├── docs/                         # Additional documentation
│   ├── api/                      # API documentation
│   ├── deployment/               # Deployment guides
│   └── development/              # Development guides
├── intent-engine/                # Backend service (current directory)
│   └── [all backend files]
├── frontend/                     # Frontend dashboard (planned)
│   ├── package.json
│   ├── src/
│   └── public/
├── infrastructure/               # Infrastructure as Code
│   ├── terraform/                # Terraform configurations
│   ├── kubernetes/               # K8s manifests
│   └── ansible/                  # Ansible playbooks
├── scripts/                      # Root-level scripts
│   ├── setup.sh                  # Initial setup script
│   ├── deploy.sh                 # Deployment script
│   └── backup.sh                 # Backup script
└── tests/                        # End-to-end tests
    ├── e2e/                      # End-to-end test suites
    └── integration/              # Integration tests
```

## Files to Copy to Parent Directory

### Essential Files

1. **README.md** - Copy from `intent-engine/README.md`
   - Update paths to reference `intent-engine/` subdirectory
   - Add frontend section when available

2. **LICENSE** - Create MIT License file
   - Use standard MIT License template
   - Add copyright year and owner

3. **.gitignore** - Copy from `intent-engine/.gitignore`
   - Add frontend ignore patterns when available
   - Add infrastructure ignore patterns

4. **docker-compose.yml** - Create root-level compose
   ```yaml
   version: '3.8'
   
   services:
     intent-engine:
       build: ./intent-engine
       ports:
         - "8000:8000"
       environment:
         - DATABASE_URL=postgresql+asyncpg://user:pass@db:5432/intent_engine
       depends_on:
         - db
         - redis
   
     db:
       image: postgres:15-alpine
       volumes:
         - postgres_data:/var/lib/postgresql/data
       environment:
         - POSTGRES_DB=intent_engine
         - POSTGRES_USER=user
         - POSTGRES_USER_PASSWORD=pass
   
     redis:
       image: redis:7-alpine
       volumes:
         - redis_data:/data
   
   volumes:
     postgres_data:
     redis_data:
   ```

### Documentation Files

Create `docs/` directory with:

1. **docs/README.md** - Documentation index
2. **docs/QUICKSTART.md** - Quick start guide
3. **docs/ARCHITECTURE.md** - Architecture overview
4. **docs/API_REFERENCE.md** - API documentation
5. **docs/DEPLOYMENT.md** - Deployment guide
6. **docs/CONTRIBUTING.md** - Contribution guidelines

### Scripts

Create `scripts/` directory with:

1. **scripts/setup.sh** (Linux/Mac)
   ```bash
   #!/bin/bash
   # Setup script for Intent Engine
   
   echo "Setting up Intent Engine..."
   
   # Install Python dependencies
   cd intent-engine
   pip install -r requirements.txt
   
   # Setup database
   python init_sample_data.py
   
   echo "Setup complete!"
   ```

2. **scripts/setup.bat** (Windows)
   ```batch
   @echo off
   echo Setting up Intent Engine...
   
   cd intent-engine
   pip install -r requirements.txt
   python init_sample_data.py
   
   echo Setup complete!
   ```

3. **scripts/deploy.sh** - Production deployment script
4. **scripts/backup.sh** - Database backup script

## Frontend Directory (Planned)

When the frontend is developed, create:

```
frontend/
├── package.json
├── tsconfig.json
├── README.md
├── .env.example
├── public/
│   ├── index.html
│   └── favicon.ico
└── src/
    ├── index.tsx
    ├── App.tsx
    ├── components/
    │   ├── Dashboard.tsx
    │   ├── Campaigns.tsx
    │   ├── Ads.tsx
    │   └── Analytics.tsx
    ├── services/
    │   └── api.ts
    └── styles/
        └── index.css
```

## Infrastructure Directory (Optional)

For production deployments:

```
infrastructure/
├── terraform/
│   ├── main.tf
│   ├── variables.tf
│   └── outputs.tf
├── kubernetes/
│   ├── deployment.yaml
│   ├── service.yaml
│   └── ingress.yaml
└── ansible/
    ├── playbook.yml
    └── inventory.ini
```

## Migration Steps

To move from current structure to recommended structure:

1. **Create parent README.md**
   - Copy `intent-engine/README.md` to parent
   - Update paths and add multi-project context

2. **Create LICENSE file**
   - Add MIT License to parent directory

3. **Update .gitignore**
   - Copy to parent
   - Add patterns for all subdirectories

4. **Create root docker-compose.yml**
   - Define all services at root level
   - Reference intent-engine build context

5. **Create docs/ directory**
   - Move documentation files as appropriate
   - Create documentation index

6. **Create scripts/ directory**
   - Add setup and deployment scripts
   - Make scripts executable

7. **Update CI/CD configurations**
   - Update GitHub Actions or other CI configs
   - Reference correct paths

## Git Repository Structure

The Git repository should maintain:

```
intent-engine/ (submodule or subdirectory)
  - All backend code
  - Backend tests
  - Backend documentation

frontend/ (separate repo or subdirectory)
  - All frontend code
  - Frontend tests

Shared files in root:
  - README.md
  - LICENSE
  - .gitignore
  - docker-compose.yml
  - CI/CD configurations
```

## Notes

- Keep backend code in `intent-engine/` subdirectory
- Frontend can be separate repo or subdirectory
- Shared configuration at root level
- Documentation should be accessible from root
- Scripts should work from root directory

## Current Status

Currently, only the `intent-engine/` directory exists with the backend code.
The parent directory structure should be created as the project grows to include:
- Frontend dashboard
- Infrastructure as code
- Additional documentation
- Deployment scripts
