# SignalX Git & Commit Conventions

This document establishes the Git workflow, commit message standards, branch naming rules, and pull request conventions for the `signalx` repository.

---

## 1. Commit Message Format

SignalX enforces the following structured commit message format:

```
<branch>(<next_tag|next_version>): <short summary in lowercase>

[optional body providing detailed context, rationale, or breaking changes]

[optional footer(s) such as Closes #123, Refs #456]
```

### Components
1. **`<branch>`**: The current working branch name (e.g. `develop`, `feature/signals-expansion`, `main`).
2. **`<next_tag|next_version>`**: The upcoming target semantic release version tag, calculated as follows:
   - **Calculation Rule**: `next_tag|next_version` = **Current Tag + `0.1.0`** (minor version bump, e.g. `v0.1.0` $\rightarrow$ `v0.2.0`, `v0.2.0` $\rightarrow$ `v0.3.0`).
   - **Default / Initial State**: If the repository has no existing tags or versions yet, the default value is always **`v0.1.0`**.
3. **`<short summary in lowercase>`**: Concise imperative description of the change starting with a lowercase letter (no trailing period).

### Examples
- `develop(v0.1.0): initial release with core pipeline and base signals` *(when no prior tags exist)*
- `develop(v0.2.0): add 10 smc and candlestick signals (cdl028-cdl037)` *(when current tag is v0.1.0)*
- `develop(v0.2.0): add 12 dsp and trend signals (trd042-trd053)`
- `feature/smc-signals(v0.3.0): implement fvg mitigation and order block retest` *(when current tag is v0.2.0)*
- `main(v1.0.0): release signalx 230 quantitative signals suite`

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
