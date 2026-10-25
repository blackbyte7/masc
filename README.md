<div align="center">

# 🧠 MASC: Modular Adversarial Synergy Chain

**A General-Purpose Dialectical and Adversarial Architecture for Autonomous Agentic AI Robustness.**

</div>

<p align="center">
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT">
  </a>
  <a href="https://www.python.org">
    <img src="https://img.shields.io/badge/Python-3.10+-3776AB.svg?logo=python" alt="Python 3.10+">
  </a>
  <a href="https://python.langchain.com/docs/langgraph/">
    <img src="https://img.shields.io/badge/Built%20with-LangGraph-229540.svg" alt="Built with LangGraph">
  </a>
  <a href="https://langfuse.com/">
    <img src="https://img.shields.io/badge/Observability-Langfuse-FF2D20.svg" alt="Langfuse Observability">
  </a>
  <a href="https://modelcontextprotocol.io/">
    <img src="https://img.shields.io/badge/Protocol-MCP-8A2BE2.svg" alt="Model Context Protocol">
  </a>
</p>

---

## 📖 Overview

MASC (Modular Adversarial Synergy Chain) is a framework designed to force AI-generated or existing artifacts to withstand
rigorous, multi-vector criticism before finalization. Moving beyond linear generation pipelines, MASC implements a
structured, internal dialectical debate (Thesis ➔ Antithesis ➔ Synthesis).

Whether generating code, strategic proposals, or analytical reports from scratch—or auditing existing files in
**Audit-Only Mode** (`initial_artifact`)—MASC surrounds the artifact with specialized adversarial agents (e.g.,
`CodeAuditor`, `DA`, `UncertaintyQuantifier`, `SecurityHawk`, or request-scoped inline personas) that ruthlessly critique
the artifact against the original task description. A Synthesizer then rebuilds the artifact to patch vulnerabilities,
looping over multiple cycles with configurable early convergence, token budget circuit breakers, and optional
Human-in-the-Loop (HITL) approval breakpoints.

## ✨ Key Features

* **Dialectical & Audit-Only Engine:** Execute full Propose ➔ Critique ➔ Synthesize cycles managed by LangGraph, or supply an `initial_artifact` to skip the Proposer and audit existing code, policies, or architecture docs directly.
* **Universal Entrypoints:** Access the engine via a rich Gradio Studio UI, a FastAPI SSE server, an OpenAI-compatible `/v1/chat/completions` API, a headless CLI, or as a native **Model Context Protocol (MCP)** server (`stdio`).
* **Static, Custom & Inline Personas:** Use built-in critics, load custom personas from `custom_personas.json`, pass ephemeral `critique_type` and `system_prompt` definitions inline per request, or enable the optional **Auto-Persona Router** (`auto_select_adversaries`).
* **Deterministic Verification Grounding:** Optionally run deterministic static checks (`enable_verification_tools`) including Python AST syntax and dangerous call analysis (`eval`, `exec`, `shell=True`, unsafe deserialization), hardcoded secret scanning, and JSON validation to ground adversary critiques.
* **Human-in-the-Loop (HITL) Breakpoints:** Pause graph execution prior to synthesis (`require_human_approval`), inspect or curate the pending `CritiquesCollection`, and resume or abort via API, UI, CLI, or MCP.
* **FinOps & Convergence Governance:** Track cumulative `TokenUsage` (`input_tokens`, `output_tokens`, `total_tokens`, `llm_calls`), enforce `max_total_tokens` circuit breakers, configure `stop_on_max_severity` early convergence thresholds, and enforce structured JSON output (`response_schema`).
* **Production Security & Resiliency:** Granular per-node LLM routing (OpenAI, Anthropic, Google, Ollama) hardened with `base_url` SSRF allowlisting, API key exfiltration guards, constant-time bearer auth (`MASC_API_KEY`), global LLM concurrency semaphores, sliding-window IP rate limiting, and SSE `: keep-alive` heartbeats.

---

## 🧠 Dual Synthesis Protocols

The true power of MASC lies in how it intelligently resolves conflicting adversarial feedback. The Synthesis Node is a
pluggable module that supports two distinct synthesis strategies, allowing you to select the optimal protocol based on
the specific requirements of the task.

### Sequential Refinement (The Iterative Path) - *Default*
This protocol is a robust choice for technical tasks, code generation, and processes requiring high reliability and a
clear, auditable trail of changes. It operates as a stateful, iterative loop, deterministically triaging critiques
(`ANTAGONISTIC` first, followed by `CRITICAL`, `HIGH`, `MEDIUM`, and `LOW` severity `CONSTRUCTIVE` critiques) and
resolving conflicts while anchored to the original task description and preserving fixes applied earlier in the same turn.
* **Best For:** Verifiable domains (like code or data pipelines), tasks demanding high security or accuracy, and scenarios where traceability of changes is paramount.
* **Batched Constructive Option (`batch_constructive_critiques`):** When enabled, Sequential Refinement refutes antagonistic challenges first and then applies all triaged constructive critiques in a single unified pass per turn, cutting latency and token usage by up to 60%.

### The Architect Protocol (Holistic Integration)
This protocol is a holistic strategy designed for high-stakes creative or strategic tasks where a globally optimal
solution is desired. It integrates all constructive feedback and refutes antagonistic challenges in a planned and
unified manner.
* **Best For:** Creative or strategic tasks, non-verifiable domains, and scenarios where novel synthesis is valued over strict traceability.

**Configuration Options:**
You can easily toggle between these protocols depending on your entrypoint:
* **Gradio UI:** Use the "Synthesis Protocol" radio button under the "Core Roles & Settings" tab.
* **CLI / API / MCP:** Set the `"synthesis_protocol"` key in your JSON configuration to either `"Sequential Refinement"` or `"Architect"`.

---

## 🛠️ Installation & Setup

MASC runs out-of-the-box using a bounded in-memory checkpointer (`memory://`) and can operate entirely on local models
(via Ollama) without any API keys. For production deployments requiring long-term state persistence across sessions, an
optional PostgreSQL database is supported. Docker Compose is the recommended way to run the entire suite cleanly.

### 1. Clone the Repository

```bash
git clone [https://github.com/your-username/masc.git](https://github.com/your-username/masc.git)
cd masc

```

### 2. Configure Environment Variables

Create a `.env` file in the root directory. MASC will automatically route to the correct provider based on your workflow
configuration.

```env
# -----------------------------
# LLM Provider Keys
# -----------------------------
OPENAI_API_KEY="sk-proj-..."
ANTHROPIC_API_KEY="sk-ant-..."
GOOGLE_API_KEY="AIza..."

# -----------------------------
# Security & Rate Limiting (Optional)
# -----------------------------
# Protects both the FastAPI server (Bearer token) and Gradio Studio UI (login password)
# MASC_API_KEY="your-production-bearer-token"
# RATE_LIMIT_PER_MINUTE=60
# MAX_CONCURRENT_LLM_CALLS=10
# ALLOWED_BASE_URLS='["http://localhost:11434","[http://127.0.0.1:11434](http://127.0.0.1:11434)","[http://host.docker.internal:11434](http://host.docker.internal:11434)"]'
# ALLOW_CUSTOM_BASE_URLS=false

# -----------------------------
# Observability (Optional)
# -----------------------------
LANGFUSE_PUBLIC_KEY="pk-lf-..."
LANGFUSE_SECRET_KEY="sk-lf-..."
# Use [https://us.cloud.langfuse.com](https://us.cloud.langfuse.com) if using US data residency
LANGFUSE_HOST="[https://cloud.langfuse.com](https://cloud.langfuse.com)"

# -----------------------------
# State Persistence
# -----------------------------
# Defaults to memory:// outside of Docker. Docker Compose configures PostgreSQL automatically.
# DATABASE_URL="postgresql://masc_user:masc_pass@localhost:5432/masc_db"

```

### 3. Build and Start the Infrastructure

Spin up PostgreSQL, the FastAPI server (`masc-api`), and the Gradio Studio (`masc-ui`) in detached mode:

```bash
docker compose up -d

```

---

## 🚀 Usage Guide

MASC uses a unified launcher (`main.py`) to expose its four distinct entrypoints.

### 1. Gradio Studio (UI Mode)

The easiest way to interact with MASC, configure personas, paste existing artifacts for auditing, and watch the
dialectical process unfold in real-time (including interactive Human-in-the-Loop approval controls).

```bash
# If running locally without Docker:
python main.py --mode ui

```

*Navigate to `http://127.0.0.1:7860` in your browser. If `MASC_API_KEY` is configured, log in with any username and your `MASC_API_KEY` as the password.*

### 2. API Server (Native SSE & OpenAI-Compatible)

Deploys a FastAPI instance capable of streaming graph execution states via Server-Sent Events (SSE) with automatic
keep-alive heartbeats, thread state inspection, HITL resumption, and drop-in OpenAI SDK compatibility.

```bash
# If running locally without Docker:
python main.py --mode api

```

*The server runs on `http://0.0.0.0:8000`.*

* **Native MASC Endpoints:**
* `POST /v1/masc/execute` — Execute or audit an artifact (`task_description`, `config`, optional `initial_artifact`, optional `thread_id`) via SSE stream.
* `GET /v1/masc/threads/{thread_id}` — Inspect checkpointed thread state and `awaiting_human_approval` status.
* `POST /v1/masc/threads/{thread_id}/resume` — Approve, reject, or override `critiques_collection` on a paused HITL thread and stream the resumed synthesis.
* `GET /health` — Health check probe.


* **OpenAI-Compatible Endpoints (for Cursor, Continue, OpenWebUI, LangChain):**
* `GET /v1/models` — Lists virtual preset models:
* `masc-sequential-default` (`DA` + `UncertaintyQuantifier`)
* `masc-architect-default` (`DA` + `ContextualChallenger`)
* `masc-security-audit` (`SecurityHawk` + `CodeAuditor` + static verification tools)
* `masc-auto` (Auto-Persona Router + static verification tools)


* `POST /v1/chat/completions` — Standard OpenAI chat completions (supports both `stream=false` JSON and `stream=true` SSE chunks, plus `stream_options.include_usage`).



### 3. Command Line Interface (CLI Mode)

Run MASC headlessly for CI/CD pipelines, interactive code reviews, or automated batch processing.

First, generate a base configuration file:

```bash
python main.py --mode cli --generate-template config.json

```

Execute a workflow from scratch:

```bash
python main.py --mode cli --task "Draft a migration strategy to microservices." --config config.json

```

Audit an existing file directly (skipping the Proposer) with the optional Auto-Persona Router and Deterministic Verification:

```bash
python main.py --mode cli \
  --task "Audit this service for SQL injection and IAM flaws." \
  --config config.json \
  --initial-artifact service.py \
  --auto-personas \
  --verify-tools \
  --approve-all

```

### 4. Model Context Protocol (MCP Mode) via Docker

You can expose MASC as an adversarial auditing tool to external autonomous agents (like Claude Desktop, Cursor, or
LangChain agents) over `stdio`. Two MCP tools are registered:

* `trigger_masc_audit(task_description, config_json, initial_artifact=None)`
* `resume_masc_audit(thread_id, approve=True, critiques_override_json=None)`

**Adding to Claude Desktop (`claude_desktop_config.json`):**

```json
{
  "mcpServers": {
    "MASC-Dialectics": {
      "command": "docker",
      "args": [
        "compose",
        "-f",
        "/absolute/path/to/masc/docker-compose.yml",
        "run",
        "--rm",
        "-i",
        "masc-api",
        "python",
        "main.py",
        "--mode",
        "mcp"
      ]
    }
  }
}

```

*Note: Replace `/absolute/path/to/masc/docker-compose.yml` with the actual path to your cloned directory.*

---

## ⚙️ Advanced `MASCConfig` Options

In addition to `proposer`, `synthesizer`, `adversaries`, `synthesis_protocol`, and `max_turns`, `MASCConfig` supports the
following optional fields:

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `initial_artifact` | `Optional[str]` | `null` | Existing text/code to audit directly, skipping the `propose` node. |
| `auto_select_adversaries` | `bool` | `false` | Optional router that dynamically selects 1–3 relevant personas from the registry based on the task. |
| `stop_on_max_severity` | `"CRITICAL" | "HIGH" | "MEDIUM" | "LOW" | "NONE"` | `"LOW"` | Minimum critique severity required to continue looping across turns. |
| `batch_constructive_critiques` | `bool` | `false` | Merges all constructive critiques into a single synthesis pass per turn in Sequential mode. |
| `enable_verification_tools` | `bool` | `false` | Runs deterministic Python AST, JSON, and secret-pattern checks and injects findings into adversary prompts. |
| `require_human_approval` | `bool` | `false` | Pauses execution before synthesis (`__interrupt__`) until resumed via `/resume`, UI, CLI, or MCP. |
| `response_schema` | `Optional[dict]` | `null` | JSON Schema enforced on the final turn's synthesized output. |
| `max_total_tokens` | `Optional[int]` | `null` | Cumulative token budget across the run; halts further turns when reached. |

---

## 🦙 Using Local Models (Ollama)

MASC fully supports running your workflows using entirely local, open-weight models via Ollama (e.g., `deepseek-r1`,
`llama-3.3`, `mistral`, `qwen2.5`).

**Important Networking Note for Docker Users:**
If you are running MASC via Docker Compose while Ollama runs natively on your host machine, configure:

* **Provider:** `Ollama`
* **Base URL:** `http://host.docker.internal:11434` (or `http://localhost:11434` when running MASC natively outside Docker)

---

## 🎭 Custom & Inline Personas Architecture

MASC supports two ways to define custom adversarial critics without modifying Python code:

### 1. Persistent Custom Personas (`custom_personas.json`)

Create `custom_personas.json` in the project root (or set `MASC_PERSONAS_FILE`):

```json
{
  "ComplianceOfficer": {
    "critique_type": "CONSTRUCTIVE",
    "system_prompt": "Assume the role of a strict Regulatory Compliance Officer. Your function is to audit the proposal for global regulatory violations (e.g., GDPR, CCPA, HIPAA). Provide your output as a list of critique items with 'severity' ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW'), 'description', and 'recommendation' fields."
  },
  "Nihilist": {
    "critique_type": "ANTAGONISTIC",
    "system_prompt": "Act as a pure antagonist. Identify the core business value claimed by the text and formulate a devastating argument as to why the endeavor is financially ruinous. Do not offer solutions. Your output is designed to be refuted, not integrated. Provide your output as a list of critique items with 'severity' (always 'HIGH'), 'description', and 'recommendation' fields."
  }
}

```

### 2. Ephemeral Inline Personas (Per-Request)

In API, CLI, or MCP configurations, any entry in `adversaries` can supply `critique_type` and `system_prompt` inline:

```json
{
  "persona_name": "EphemeralFinOpsAuditor",
  "critique_type": "CONSTRUCTIVE",
  "system_prompt": "Audit the proposed cloud architecture strictly for cross-AZ data transfer and NAT gateway cost risks.",
  "llm_config": {
    "provider": "OpenAI",
    "model_name": "gpt-5.2-pro",
    "temperature": 0.1,
    "max_tokens": 4096
  }
}

```

---

## 🤝 Pairing Personas with Synthesis Protocols

* **For the Sequential Protocol (Verifiable Tasks):** Map adversaries to the technical nature of the task. Prioritize technical auditors like `CodeAuditor`, `SecurityHawk`, or custom personas like `ZeroTrustArchitect`, and enable `enable_verification_tools: true`.
* **For the Architect Protocol (Strategic/Creative Tasks):** Analytical or strategic tasks demand epistemic and contextual critics. Use personas like `UncertaintyQuantifier`, `ContextualChallenger`, or a custom `UXPessimist`.
* **Universal Hardening:** For any high-stakes artifact where the defensibility of its core argument is paramount, include a pure antagonist like `DA` (or `NihilisticDetractor`) regardless of the chosen protocol.

---

## ⚖️ License

This project is licensed under the MIT License. See the LICENSE file for the full text.