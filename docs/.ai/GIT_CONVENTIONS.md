# SignalX Git & Commit Conventions

This document establishes the Git workflow, commit message standards, branch naming rules, and pull request conventions for the `signalx` repository.

---

## 1. Commit Message Format (Conventional Commits)

SignalX strictly enforces the [Conventional Commits v1.0.0](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <short summary in lowercase>

[optional body providing detailed context, rationale, or breaking changes]

[optional footer(s) such as Closes #123, Refs #456]
```

### Allowed Types
| Type | Description | Example |
| :--- | :--- | :--- |
| `feat` | Adding a new signal, indicator, CLI command, or feature | `feat(signals): add 10 SMC and candlestick signals (CDL028-CDL037)` |
| `fix` | Fixing a bug, calculation error, or edge case | `fix(volatility): prevent division by zero in NATR stretch calculation` |
| `perf` | Optimizing execution speed, memory footprint, or vectorization | `perf(core): vectorize master ensemble consensus calculation` |
| `test` | Adding or updating unit/integration tests | `test(statistical): add test cases for FDI and variance ratio` |
| `docs` | Documentation changes, catalog updates, docstrings | `docs(catalog): update signals catalog to 230 signals` |
| `refactor` | Code restructuring without changing external behavior | `refactor(trend): extract shared moving average helper` |
| `style` | Formatting, whitespace, imports sorting (`ruff format`) | `style: format signals modules according to ruff` |
| `chore` | Build scripts, dependencies, CI/CD, repository configuration | `chore(deps): bump pandas-ta and scipy dependencies` |

### Scopes
Recommended scopes to keep git history modular:
- `signals` (General signal engine changes)
- `candlestick`, `trend`, `volume`, `volatility`, `momentum`, `statistical`, `composite` (Category modules)
- `core` (Signal orchestration, `generate_signals`)
- `metadata` (Signal catalog, code/name mapping)
- `cli` (Command-line interface commands)
- `utils` (Normalization, I/O, statistics)
- `catalog` (Documentation catalog)

### Rules & Formatting
- **Subject line length**: Maximum 72 characters.
- **Tense & Mood**: Use imperative present tense ("add", "fix", "update", NOT "added", "fixing").
- **Case**: Summary starts with a lowercase letter (e.g. `feat(volume): implement klinger volume oscillator`).
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
