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

MASC (Modular Adversarial Synergy Chain) is an enterprise-grade framework designed to force AI-generated artifacts to withstand rigorous, multi-vector criticism before finalization. Moving beyond linear generation pipelines, MASC implements a structured, internal dialectical debate (Thesis -> Antithesis -> Synthesis).

Whether generating code, strategic proposals, or analytical reports, MASC surrounds the initial generation with specialized adversarial agents (e.g., `CodeAuditor`, `Devil'sAdvocate`, `UncertaintyQuantifier`) that ruthlessly critique the artifact. A Synthesizer then rebuilds the artifact to patch vulnerabilities, optionally looping over multiple cycles to produce a highly robust, hardened final output.



## ✨ Key Features

* **Dialectical Engine:** Implements a strict Propose ➔ Critique ➔ Synthesize cycle managed by LangGraph.
* **Universal Entrypoints:** Access the engine via a rich Gradio UI, a FastAPI server, a CLI, or as a native **Model Context Protocol (MCP)** tool for external agentic workflows.
* **Modular Personas:** A plug-and-play dictionary of specialized critics. Easily define custom adversaries to target domain-specific blind spots.
* **Granular LLM Routing:** Mix and match providers (OpenAI, Anthropic, Google, local Ollama) on a per-node basis. Run the Proposer on Claude 3.5 Sonnet, the Adversaries on GPT-4o, and the Synthesizer on Llama-3.
* **Production Observability:** Native, zero-config Langfuse integration for tracing node latency, token usage, and step-by-step reasoning.
* **Enterprise State Management:** Built-in PostgreSQL checkpointer support for persistent state across complex, long-running graphs.

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

## 🔬 Extending the Framework (Custom Personas)

You can easily expand MASC's library of adversarial critics without modifying any Python code.

To add custom personas, simply create a file named `custom_personas.json` in the root directory of the project (alongside `main.py`). When the application starts, it will automatically load these and merge them with the default personas. They will instantly appear in the Gradio UI dropdowns and be accessible via the CLI and MCP endpoints.

### File Format

The JSON file must consist of a dictionary where the keys are the **Persona Names** (no spaces recommended) and the values are objects containing:

* `critique_type`: Must be either `"CONSTRUCTIVE"` (aims to improve the artifact) or `"ANTAGONISTIC"` (aims to destroy the artifact's fundamental premise).
* `system_prompt`: The detailed instructions for the LLM.

**Example `custom_personas.json`:**

```json
{
  "ComplianceOfficer": {
    "critique_type": "CONSTRUCTIVE",
    "system_prompt": "Assume the role of a strict Regulatory Compliance Officer. Your function is to audit the proposal for global and US regulatory violations (e.g., GDPR, CCPA, HIPAA). Provide your output as a list of dictionary objects with 'severity', 'description', and 'recommendation' keys."
  }
}

```

Restart your environment, and the new persona will instantly be available in the UI, API, and CLI configurations.

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

This project is licensed under the MIT License. See the [LICENSE](https://www.google.com/search?q=LICENSE) file for the full text.