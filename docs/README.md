# SignalX Documentation Hub

Welcome to the `signalx` documentation index. `signalx` is a quantitative signal extraction library providing 230 standardized technical and statistical trading signals across 7 indicator families.

---

## AI Agent & Developer Documentation (`docs/.ai/`)

The core documentation is organized into modular files inside `docs/.ai/`:

| Document | Description |
| :--- | :--- |
| [AI Agent Guidelines](.ai/AI_AGENT_GUIDELINE.md) | Coding standards, commands, test performance rules, and agent workflows. |
| [Architecture Rules & Invariants](.ai/RULE.md) | Core invariants: `_signal` suffix, 4-state contract (`buy`, `sell`, `hold`, `none`), zero lookahead leakage. |
| [Repository Structure](.ai/STRUCTURE.md) | Detailed directory tree layout, module responsibilities, and source mappings. |
| [Signals Catalog](.ai/SIGNALS_CATALOG.md) | Full breakdown and trigger condition reference for all 230 trading signals. |
| [Configuration & Schemas](.ai/CONFIGURATION.md) | Input DataFrame schemas, column normalization aliases, and CLI parameters. |
| [Usage & Examples](.ai/USAGE.md) | Comprehensive Python API code examples, CLI workflows, and ML integration patterns. |

---

## Agent Guide Synchronization

The markdown documentation files in `docs/.ai/` are automatically compiled into single-file AI prompts for various AI coding assistants:

- `AGENTS.md` (General LLM Coding Agents)
- `GEMINI.md` (Google Gemini CLI / Antigravity Agent)
- `CLAUDE.md` (Anthropic Claude Code Agent)

To recompile all agent markdown files after updating documentation in `docs/.ai/`, execute:

```bash
bash scripts/generate_agents_markdown.sh
```

---

## Key Library Highlights

- **230 Standardized Signals**: Covers Trend (53), Momentum (38), Volatility (42), Volume (28), Candlestick (37), Statistical (20), Composite (12).
- **Uniform Contract**: Every signal returns strictly `"buy"`, `"sell"`, `"hold"`, or `"none"`.
- **Zero Future Leakage**: All mathematical logic is leak-free and causal ($t$ only).
- **Sub-Second Test Execution**: 100% test suite executes against synthetic test fixtures in seconds.
