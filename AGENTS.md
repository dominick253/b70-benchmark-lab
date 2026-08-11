<!-- dox-managed:v2 -->
# b70-benchmark-lab Repository Contract

## Core Contract

- This file is the root work contract for `dominick253/b70-benchmark-lab`.
- **DOX chain:** `~/.hermes/AGENTS.md` -> `/home/dom/AGENTS.md` -> this file -> nearest child `AGENTS.md`.
- Read the full chain before editing any file.
- A closer contract can add constraints but cannot weaken a parent contract.
- After each meaningful change, perform a DOX pass. Update stale commands, paths, architecture notes, and safety rules.
- Source code, tests, build files, and CI configuration are authoritative. Documentation must match them.

## Repository Mission

Enterprise GPU Benchmarking Suite for Intel Arc Pro B70.

## Repository Facts

- Default branch: `master`
- Primary toolchains: Python
- Visibility: public
- Generated and dependency directories are not source. Do not edit them unless the repository explicitly tracks them as inputs.

## Repository Map

| Path | Purpose |
|---|---|
| `benchlab/` | Repository area. Inspect its files and local documentation before editing. |
| `configs/` | Repository area. Inspect its files and local documentation before editing. |
| `results/` | Repository area. Inspect its files and local documentation before editing. |
| `tests/` | Automated tests. |
| `README.md` | Root project file or artifact. |
| `benchlab/` | Repository area. Inspect its files and local documentation before editing. |
| `configs/` | Repository area. Inspect its files and local documentation before editing. |
| `pyproject.toml` | Root project file or artifact. |
| `results/` | Repository area. Inspect its files and local documentation before editing. |
| `tests/` | Automated tests. |

## Scoped Contracts

- No child `AGENTS.md` files are present. Add one when a subtree needs stricter rules.

When work enters a scoped path, read every parent contract in order. If a listed child file is moved or deleted, update this section in the same change.

## Authoritative Commands

| Command | Purpose |
|---|---|
| `python -m compileall .` | Checks Python syntax without changing source. |
| `python -m pytest` | Runs the Python test suite. |
| `ruff check .` | Runs configured Python lint checks. |

- Use only commands supported by files in this repository.
- Inspect package scripts, Make targets, Gradle tasks, workflows, and tool configuration before adding commands.
- Do not invent a passing command or treat an unavailable tool as a successful check.

## Change Workflow

1. Read this contract and the nearest scoped contract.
2. Inspect the implementation, tests, manifests, and relevant workflow files.
3. Record the current behavior or reproduce the defect.
4. Make the smallest coherent change. Do not add unrelated cleanup.
5. Run the narrowest relevant check first.
6. Run the repository's broader build, test, lint, and type checks that apply.
7. Review the diff for generated files, secrets, unrelated edits, and stale documentation.
8. Perform a final DOX pass before commit.

## Verification Contract

- A documentation-only change must validate referenced paths, commands, links, and examples.
- A source change must pass its focused tests plus the applicable repository checks above.
- A build configuration or CI change must be verified locally when possible and then verified in CI.
- A UI change needs functional state checks, not only screenshots.
- A service change needs a bounded startup or health check and clean shutdown evidence.
- If a required check cannot run, report `NOT RUN` with the exact blocker. Never report it as passed.
- Do not claim completion while required checks fail.

## Safety and Boundaries

- Never commit credentials, tokens, private keys, device passwords, `.env` values, or production data.
- Do not modify generated output, vendored dependencies, lockfiles, or snapshots unless the task requires it and verification covers it.
- Do not run deployment, release, migration, destructive database, infrastructure, or external messaging actions without explicit task scope.
- Preserve backward compatibility unless the task explicitly authorizes a breaking change.
- Treat workflow files, authentication, payment, analytics, and data migrations as high-risk areas.
- Do not disable tests, lint rules, security checks, or CI gates to obtain a pass.

## Git and CI

- Keep commits focused and use an accurate conventional commit message when the repository accepts it.
- Inspect the full staged diff before commit.
- Do not rewrite shared history.
- After push, inspect every triggered CI job. A successful push is not proof of a successful change.
- If no CI exists, state that limitation and rely on the documented local verification.

### Discovered Workflows

- No GitHub Actions workflow was discovered.

## Documentation Freshness

Update this file in the same change when any of these change:

- repository structure or ownership boundaries;
- build, test, lint, format, type-check, deploy, or release commands;
- required environment variables or external services;
- CI jobs, quality gates, or branch policy;
- public interfaces, data contracts, security rules, or migration steps;
- child `AGENTS.md` scope.

Remove stale guidance. Do not preserve obsolete instructions as historical notes inside an active contract.
<!-- /dox-managed -->
