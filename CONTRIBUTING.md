# Contributing to MASC

Thanks for your interest in improving MASC! This guide covers the local workflow expected for pull requests.

## Development setup

```bash
git clone https://github.com/blackbyte7/masc.git
cd masc
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env               # optional: only needed to run against real providers
```

Python 3.10+ is supported; CI runs 3.10 – 3.13.

## Running checks

```bash
ruff check .            # lint
ruff format .           # auto-format (CI runs `ruff format --check .`)
pytest                  # full test suite (hermetic: never reads your .env)
pytest --cov=src        # with coverage
```

The test suite is fully offline. It uses fake chat models and the in-memory checkpointer, and it ignores
your local `.env` (see `tests/conftest.py`). Please keep it that way: new tests must not need network
access or real API keys.

## Pull request guidelines

1. Branch from `master` and keep each PR focused on one change.
2. Add or update tests for every behaviour change or bug fix.
3. Never commit secrets, `.env` files, IDE folders or generated dumps.
4. User-facing changes (new config fields, endpoints, CLI flags) need a README update.

## Project layout

| Path | Purpose |
| --- | --- |
| `src/core/engine.py` | LangGraph graph, nodes, checkpointer and workflow streaming |
| `src/core/state.py` | Pydantic models (`MASCConfig`, critiques, token usage) |
| `src/core/personas.py` | Built-in/custom personas and deterministic verification |
| `src/core/validation.py` | Shared request-level config validation used by every entrypoint |
| `src/core/credentials.py` | Ephemeral API-key vault (keys are never checkpointed) |
| `src/entrypoints/` | FastAPI server, Gradio Studio, CLI and MCP server |
| `src/config/settings.py` | Environment-driven settings and SSRF-safe `base_url` validation |
| `tests/` | Offline unit and integration tests |

## Reporting security issues

Please **do not** open public issues for vulnerabilities. Report them privately through
[GitHub Security Advisories](https://github.com/blackbyte7/masc/security/advisories/new).
