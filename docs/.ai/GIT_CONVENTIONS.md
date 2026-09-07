# SignalX Git & Commit Conventions

This document establishes the Git workflow, commit message standards, branch naming rules, and pull request conventions for the `signalx` repository.

---

## 1. Commit Message Format

SignalX enforces the following structured commit message format:

```
<branch>(v<version>): <short summary in lowercase>

[optional body providing detailed context, rationale, or breaking changes]

[optional footer(s) such as Closes #123, Refs #456]
```

### Components
1. **`<branch>`**: The current working branch name (e.g. `develop`, `feature/signals-expansion`, `main`).
2. **`v<version>`**: The exact project version defined in `pyproject.toml` (under `[project].version`, prefixed with `v`, e.g. `version = "0.1.0"` $\rightarrow$ `v0.1.0`).
   - **Source of Truth**: Always extract directly from `pyproject.toml`.
   - **No Version Guessing**: Do not calculate or predict future release versions (no `next_tag` or `next_version`). Always use the active version currently declared in `pyproject.toml`.
3. **`<short summary in lowercase>`**: Concise imperative description of the change starting with a lowercase letter (no trailing period).

### Examples
- `develop(v0.1.0): initial release with core pipeline and base signals`
- `develop(v0.1.0): add 12 mean reversion signals (mr001-mr012)`
- `feature/smc-signals(v0.1.0): implement fvg mitigation and order block retest`
- `main(v0.1.0): release signalx 251 quantitative signals suite`

### Rules & Formatting
- **Subject line length**: Maximum 72 characters.
- **Tense & Mood**: Use imperative present tense ("add", "fix", "update", NOT "added", "fixing").
- **Case**: Summary starts with a lowercase letter.
- **No trailing period**: Do not end the subject line with a period `.`.

---

## 2. Branch Naming Conventions

All branch names must be lowercase, hyphen-separated, and prefixed with the category:

```
<prefix>/<short-description>
```

### Branch Prefixes:
- `feature/<name>`: New features or signals (e.g. `feature/230-signals-expansion`, `feature/smc-fvg-signals`)
- `fix/<issue>`: Bug fixes (e.g. `fix/supertrend-warmup-nan`, `fix/cli-inspect-empty-df`)
- `perf/<target>`: Performance optimizations (e.g. `perf/vectorize-composite-voting`)
- `docs/<topic>`: Documentation enhancements (e.g. `docs/update-signals-catalog`)
- `refactor/<scope>`: Code refactoring (e.g. `refactor/modularize-signal-helpers`)

---

## 3. Workflow & Invariants

1. **Atomic Commits**: Each commit should represent a single logical change that passes all tests and linter checks.
2. **Pre-Commit Verification**:
   Before committing, always run:
   ```bash
   uv run pytest -v
   uv run ruff check .
   uv run ruff format --check .
   ```
3. **AI Documentation Sync**:
   When any file in `docs/.ai/` is created or modified, always compile the master documentation:
   ```bash
   bash scripts/generate_agents_markdown.sh
   ```
   and commit the auto-generated `AGENTS.md`, `GEMINI.md`, and `CLAUDE.md` along with the source docs.
4. **Gitignore Verification Before Commits**:
   Always check all `.gitignore` files (root `.gitignore` and any subfolder `.gitignore`) before staging or committing any file. Never force-add (`git add -f`) or commit files and directories that match `.gitignore` rules (such as `docs/superpowers`, `.superpowers`, `.venv`, cached files, or reports).
