# Contributing Guide

## Branching
- `main`: production
- `develop`: integration
- `feature/*`: scoped development branches

## Workflow
1. Create a feature branch from `develop`.
2. Add tests and documentation with each change.
3. Run `pytest -q` before opening PR.
4. Use conventional commit style (`feat:`, `fix:`, `docs:`, `chore:`).
