<div align="center">

# 🧠 MASC: Modular Adversarial Synergy Chain

**Official (POC) Python implementation of the *Modular Adversarial Synergy Chain* architecture.**

</div>

<p align="center">
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT">
  </a>
  <a href="https://www.python.org">
    <img src="https://img.shields.io/badge/Python-3.10+-3776AB.svg?logo=python" alt="Python 3.9+">
  </a>
  <a href="https://python.langchain.com/v0.2/docs/langgraph/">
    <img src="https://img.shields.io/badge/Built%20with-LangGraph-229540.svg" alt="Built with LangGraph">
  </a>
</p>


---

## 📖 Overview

This repository contains the official reference implementation for the MASC framework, as detailed in the white paper: **"[MASC: A General-Purpose Dialectical and Adversarial Architecture for Autonomous Agentic Robustness](LINK_TO_YOUR_PAPER.pdf)"** by Alexander Schneider (Bielefeld, 2025).

The project's goal is to provide a clear, functional, and extensible tool for exploring dialectical methods in AI quality assurance. It moves beyond linear generation pipelines to a model of structured, internal debate, forcing an AI-generated artifact to withstand multi-vector criticism before it is finalized.

## ✨ Key Features

*   **Dialectical Process:** Implements the core Thesis-Antithesis-Synthesis loop for systematic artifact hardening.
*   **Modular Personas:** A plug-and-play library of adversarial and constructive critics (`CodeAuditor`, `Devil'sAdvocate`, etc.) allows the process to be tailored to any domain.
*   **Fine-Grained Control:** The UI allows for assigning specific LLM providers (OpenAI, Anthropic, Google), models, and parameters to *each individual step* of the cognitive workflow.
*   **Streaming Interface:** The Gradio UI streams results in real-time, providing a transparent view into the agent's multi-stage reasoning process as it happens.
*   **Modern Architecture:** Built from the ground up using the latest LangChain v0.2 standards, including LangGraph for stateful agent execution.


## ⚙️ How It Works

The framework is implemented as a Directed Acyclic Graph (DAG) that manages the flow of information between specialized nodes. This graph structure is defined in `masc_t_engine_advanced.py` and orchestrated by LangGraph. The process follows four distinct stages: Propose, Adversarial Analysis, Synthesize, and producing the Final Artifact.

## 🛠️ Getting Started

### 1. Prerequisites

-   Python 3.9 or higher.
-   API keys for the LLM providers you intend to use (OpenAI, Anthropic, Google).

### 2. Installation

First, clone the repository to your local machine:
```bash
git clone https://github.com/blackbyte7/masc.git
cd masc```

Next, install the required Python packages:
```bash
pip install -r requirements.txt
```

### 3. Configuration

For convenience, you can store your API keys in a `.env` file in the project's root directory. The application will load these automatically if the API key fields in the UI are left blank.

Create a file named `.env`:
```
OPENAI_API_KEY="sk-..."
ANTHROPIC_API_KEY="sk-ant-..."
GOOGLE_API_KEY="..."
```
*(This file is listed in `.gitignore` and will not be tracked by Git.)*

### 4. Running the Application

Launch the Gradio web interface with the following command:
```bash
python app_advanced.py
```
Navigate to the local URL displayed in your terminal (e.g., `http://127.0.0.1:7860`) to access the application.

## 🔬 Extending the Framework

A primary goal of this implementation is to facilitate further research. You can easily add new adversarial or constructive personas to the framework.

1.  Open `masc_t_engine_advanced.py`.
2.  Locate the `ADVERSARY_PERSONAS` dictionary.
3.  Add a new entry with a unique name, a `critique_type`, and a detailed `system_prompt`.

**Example: Adding a "Historical Analogist" persona:**

```python
'HistoricalAnalogist': {
    "critique_type": "CONSTRUCTIVE",
    "system_prompt": "Assume the role of a Historical Analogist. Your function is to analyze the proposed strategy or artifact and identify relevant historical precedents, both successful and failed. Critique the proposal based on lessons learned from these analogies. Provide your output as a list of dictionary objects..."
}
```
After saving the file, relaunch the application. The "HistoricalAnalogist" will now appear as a selectable persona in the UI.

## 📄 How to Cite

If you use this work in your research, please cite the original paper.

```bibtex
@techreport{Schneider2025MASCT,
  author      = {Alexander Schneider},
  title       = {MASC: A General-Purpose Dialectical and Adversarial Architecture for Autonomous Agentic Robustness},
  institution = {Bielefeld},
  year        = {2025}
}
```

## ⚖️ License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for the full text.
