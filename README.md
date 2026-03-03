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

MASC (Modular Adversarial Synergy Chain) is a framework designed to force AI-generated artifacts to withstand rigorous, multi-vector criticism before finalization. Moving beyond linear generation pipelines, MASC implements a structured, internal dialectical debate (Thesis -> Antithesis -> Synthesis).

Whether generating code, strategic proposals, or analytical reports, MASC surrounds the initial generation with specialized adversarial agents (e.g., `CodeAuditor`, `Devil'sAdvocate`, `UncertaintyQuantifier`) that ruthlessly critique the artifact. A Synthesizer then rebuilds the artifact to patch vulnerabilities, optionally looping over multiple cycles to produce a highly robust, hardened final output.

## ✨ Key Features

* **Dialectical Engine:** Implements a strict Propose ➔ Critique ➔ Synthesize cycle managed by LangGraph.
* **Universal Entrypoints:** Access the engine via a rich Gradio UI, a FastAPI server, a CLI, or as a native **Model Context Protocol (MCP)** tool for external agentic workflows.
* **Modular Personas:** A plug-and-play dictionary of specialized critics. Easily define custom adversaries to target domain-specific blind spots.
* **Granular LLM Routing:** Mix and match providers (OpenAI, Anthropic, Google, local Ollama) on a per-node basis. Run the Proposer on Claude 3.5 Sonnet, the Adversaries on GPT-4o, and the Synthesizer on Llama-3.
* **Observability:** Native, zero-config Langfuse integration for tracing node latency, token usage, and step-by-step reasoning.
* **State Management:** Built-in PostgreSQL checkpointer support for persistent state across complex, long-running graphs.

---

## 🛠️ Installation & Setup

MASC requires a PostgreSQL database for state persistence and several API keys for its modular adversaries. Docker is the recommended way to run the entire suite cleanly.

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/masc.git](https://github.com/your-username/masc.git)
cd masc

```

### 2. Configure Environment Variables

Create a `.env` file in the root directory. MASC will automatically route to the correct provider based on your workflow configuration.

```env
# -----------------------------
# LLM Provider Keys
# -----------------------------
OPENAI_API_KEY="sk-proj-..."
ANTHROPIC_API_KEY="sk-ant-..."
GOOGLE_API_KEY="AIza..."

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
# Required if running outside of Docker. Docker Compose handles this automatically.
# DATABASE_URL="postgresql://user:password@localhost:5432/masc_db"

```

### 3. Build and Start the Infrastructure

Spin up the database and the backend services in detached mode:

```bash
docker compose up -d

```

---

## 🚀 Usage Guide

MASC uses a unified launcher (`main.py`) to expose its four distinct entrypoints.

### 1. Gradio Studio (UI Mode)

The easiest way to interact with MASC, configure personas, and watch the dialectical process unfold in real-time.

```bash
# If running locally without Docker:
python main.py --mode ui

```

*Navigate to `http://127.0.0.1:7860` in your browser.*

### 2. API Server

Deploys a FastAPI instance capable of streaming graph execution states via Server-Sent Events (SSE). Ideal for integrating MASC into existing web applications.

```bash
# If running locally without Docker:
python main.py --mode api

```

*The server will run on `http://0.0.0.0:8000`. Endpoint: `POST /v1/masc/execute`.*

### 3. Command Line Interface (CLI Mode)

Run MASC headlessly for CI/CD pipelines or automated batch processing.

First, generate a base configuration file:

```bash
python main.py --mode cli --generate-template config.json

```

Then, execute a workflow:

```bash
python main.py --mode cli --task "Draft a migration strategy to microservices." --config config.json

```

### 4. Model Context Protocol (MCP Mode) via Docker

You can expose MASC as a powerful, adversarial tool to external autonomous agents (like Claude Desktop, Cursor, or LangChain agents). Because MASC requires a database and complex dependencies, the cleanest way to connect it is by pointing your MCP client to the Docker container.

The external agent provides the task and configuration, while the isolated container handles the multi-agent dialectic securely.

**Adding to Claude Desktop (`claude_desktop_config.json`):**

To connect Claude Desktop to your running MASC instance, add the following configuration. By using `docker compose run`, the MCP process automatically connects to the PostgreSQL database network and inherits your `.env` keys.

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

*Note: Be sure to replace `/absolute/path/to/masc/docker-compose.yml` with the actual path to your cloned directory.*

---

## 🦙 Using Local Models (Ollama)

MASC fully supports running your workflows using entirely local, open-weight models via Ollama (e.g., `gpt-oss-20b`, `gemma-3`, `deepseek-r1`).

**Important Networking Note for Docker Users:**
If you are running MASC via Docker Compose but your Ollama instance is running natively on your host machine, `localhost` will not work. You must tell the MASC container how to reach your host network.

When configuring your provider in the UI or CLI:

* **Provider:** Select `Ollama`
* **Base URL:** Enter `http://host.docker.internal:11434` (Do **not** use `http://localhost:11434`)

Our `docker-compose.yml` is pre-configured with `host-gateway` mapping, meaning this URL will work seamlessly across Windows, macOS, and Linux.

---

## 🎭 Custom Personas Architecture

MASC is built around a plug-and-play architectural pattern for specialized adversarial agents. You are not limited to the default personas (e.g., `Devil's Advocate`, `Code Auditor`). You can inject custom critics tailored to your specific domain without modifying a single line of Python.

### How it Works

At runtime, MASC's engine dynamically loads and merges personas. By creating a `custom_personas.json` file in the root directory, you override the engine's internal dictionary.

**1. Create your definition:**
Create `custom_personas.json` next to `main.py`:

```json
{
  "ComplianceOfficer": {
    "critique_type": "CONSTRUCTIVE",
    "system_prompt": "Assume the role of a strict Regulatory Compliance Officer. Your function is to audit the proposal for global regulatory violations (e.g., GDPR, CCPA, HIPAA). Provide your output as a list of dictionary objects with 'severity', 'description', and 'recommendation' keys."
  },
  "Nihilist": {
    "critique_type": "ANTAGONISTIC",
    "system_prompt": "Act as a pure antagonist. Identify the core business value claimed by the text and formulate a devastating argument as to why the endeavor is financially ruinous. Do not offer solutions. Your output is designed to be refuted, not integrated."
  }
}

```

**2. Restart MASC:**
Upon restart, MASC will validate your schema. Valid personas will instantly populate in the Gradio UI dropdowns, the CLI template generator, and the MCP server configuration.

**Note on `critique_type`:**

* `"CONSTRUCTIVE"`: The Synthesizer agent will attempt to surgically patch the artifact based on the provided recommendations.
* `"ANTAGONISTIC"`: The Synthesizer agent will ignore the recommendations and instead rewrite the core proposal to explicitly defend against the underlying ideological challenge.

---

## 📄 How to Cite

If you use this work in your research, please cite the original paper:

```bibtex
@techreport{Schneider2025MASCT,
  author      = {Alexander Schneider},
  title       = {MASC: A General-Purpose Dialectical and Adversarial Architecture for Autonomous Agentic Robustness},
  institution = {Bielefeld},
  year        = {2025}
}

```

## ⚖️ License

This project is licensed under the MIT License. See the LICENSE file for the full text.