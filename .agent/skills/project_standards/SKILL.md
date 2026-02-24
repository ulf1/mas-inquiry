---
name: project_standards
description: Repository-wide standards for structure, documentation, version control, and CI/CD.
---

# Project Standards

This skill defines the mandatory repository structure, documentation requirements, and development workflows.

## 1. Project Structure

- `/src` — Application source code
- `/tests` — Unit and integration tests
- `/config` — Configuration files
- `/docs` — Documentation
- `/scripts` — Automation scripts

**Guideline:** Follow the established folder structure and naming conventions strictly.

## 2. Documentation

- **Public APIs:** Must include documentation/comments explaining intent and usage.
- **Complex Logic:** Document "why", not just "how".
- **Maintenance:** Update README when introducing new features.

### README Requirements
Keep `README.md` up to date with:
- Project description
- Setup instructions
- Usage examples
- Dependencies & License

## 3. Version Control (Git)

### Commit Messages
- **Format:** `<type>: <subject>` (e.g., `feat: Add user authentication`)
- **Tense:** Present tense (e.g., "Add feature" not "Added feature")
- **Types:** `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

### Branch Strategy
- `main`/`master`: Production-ready code
- `develop`: Integration branch
- `feature/*`: Feature development
- `bugfix/*`: Bug fixes
- `hotfix/*`: Critical production fixes

## 4. CI/CD Standards

- **Automation:** Automate tests, linting, and code quality checks in the pipeline.
- **Security:** Scan dependencies for known vulnerabilities and hardcoded secrets.
- **Deployment:** Automate deployments where possible.
