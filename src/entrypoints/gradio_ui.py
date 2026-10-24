import logging
import secrets
import uuid
from contextlib import asynccontextmanager

import gradio as gr
from fastapi import FastAPI

from src.config.settings import settings, validate_base_url
from src.core.engine import close_pg_pool, execute_masc_workflow
from src.core.personas import ADVERSARY_PERSONAS, load_personas
from src.core.state import LLMConfig, MASCConfig, PersonaConfig
from src.utils.model_fetcher import FALLBACK_MODELS, fetch_available_models

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def verify_gradio_auth(username: str, password: str) -> bool:
    """Authenticates Gradio UI sessions against MASC_API_KEY using constant-time comparison."""
    if not settings.masc_api_key:
        return True
    return secrets.compare_digest(password.encode("utf-8"), settings.masc_api_key.encode("utf-8"))


async def run_masc_t_cycle(task_desc, initial_artifact_text, *config_inputs, progress=gr.Progress(track_tqdm=True)):
    """Orchestrates the MASC process as an async generator, updating Gradio UI."""
    try:
        if not task_desc or not task_desc.strip():
            raise gr.Error("Task Description cannot be empty.")
        if len(task_desc) > settings.max_task_length:
            raise gr.Error(f"Task Description exceeds maximum length ({settings.max_task_length} characters).")

        cleaned_initial_artifact = (initial_artifact_text or "").strip() or None
        if cleaned_initial_artifact and len(cleaned_initial_artifact) > settings.max_task_length:
            raise gr.Error(f"Initial Artifact exceeds maximum length ({settings.max_task_length} characters).")

        progress(0, desc="Parsing workflow configuration...")

        # 1. Unpack Proposer (Index 0 to 5)
        proposer_base_url = validate_base_url(config_inputs[2], settings) if config_inputs[2] else None
        proposer_llm = LLMConfig(
            provider=config_inputs[0],
            api_key=config_inputs[1] or "",
            base_url=proposer_base_url,
            model_name=config_inputs[3],
            temperature=config_inputs[4],
            max_tokens=config_inputs[5],
        )
        proposer_cfg = PersonaConfig(persona_name="Proposer", llm_config=proposer_llm)

        # 2. Unpack Synthesizer (Index 6 to 11)
        synth_base_url = validate_base_url(config_inputs[8], settings) if config_inputs[8] else None
        synth_llm = LLMConfig(
            provider=config_inputs[6],
            api_key=config_inputs[7] or "",
            base_url=synth_base_url,
            model_name=config_inputs[9],
            temperature=config_inputs[10],
            max_tokens=config_inputs[11],
        )
        synthesizer_cfg = PersonaConfig(persona_name="Synthesizer", llm_config=synth_llm)

        synthesis_protocol_input = config_inputs[12]
        max_turns_input = int(config_inputs[13])
        stop_severity_input = config_inputs[14]
        batch_constructive_input = bool(config_inputs[15])

        # 3. Unpack Adversaries dynamically (Chunks of 7 starting at index 16)
        adversary_cfgs = []
        adv_inputs = config_inputs[16:]

        for i in range(settings.max_adversaries):
            chunk = adv_inputs[i * 7 : (i + 1) * 7]
            if chunk[0] != "NONE":
                adv_base_url = validate_base_url(chunk[3], settings) if chunk[3] else None
                adv_llm_cfg = LLMConfig(
                    provider=chunk[1],
                    api_key=chunk[2] or "",
                    base_url=adv_base_url,
                    model_name=chunk[4],
                    temperature=chunk[5],
                    max_tokens=chunk[6],
                )
                adversary_cfgs.append(PersonaConfig(persona_name=chunk[0], llm_config=adv_llm_cfg))

        if not adversary_cfgs:
            raise gr.Error("At least one Adversary must be configured.")

        masc_config = MASCConfig(
            proposer=proposer_cfg,
            synthesizer=synthesizer_cfg,
            adversaries=adversary_cfgs,
            synthesis_protocol=synthesis_protocol_input,
            max_turns=max_turns_input,
            stop_on_max_severity=stop_severity_input,
            batch_constructive_critiques=batch_constructive_input,
            initial_artifact=cleaned_initial_artifact,
        )

        progress(0.05, desc="Compiling MASC Graph & Connecting to Telemetry...")
        thread_id = str(uuid.uuid4())

        # Initialize empty states for the UI
        initial_proposal_content = cleaned_initial_artifact or ""
        critiques_dict = {"status": "Awaiting adversarial analysis..."}
        final_synthesis_content = ""
        history_md = "*Workflow initialized...*"

        async for state in execute_masc_workflow(
            task_desc, masc_config, thread_id, initial_artifact=cleaned_initial_artifact
        ):
            node_name = list(state.keys())[0]
            node_state = state[node_name]
            raw_turn = node_state.get("current_turn", 1)
            active_turn = (
                raw_turn - 1
                if node_name in ("synthesize_sequential", "synthesize_architect") and raw_turn > 1
                else raw_turn
            )

            base_progress = (active_turn - 1) / max_turns_input
            turn_weight = 1.0 / max_turns_input

            if "propose" in state:
                progress(
                    min(0.99, base_progress + (turn_weight * 0.2)),
                    desc=f"Cycle {active_turn}: Initial Proposal Generated.",
                )

                initial_proposal_content = state["propose"]["current_artifact"].content
                history = state["propose"]["current_artifact"].history
                history_md = "\n".join([f"- {entry}" for entry in history])

                yield initial_proposal_content, critiques_dict, final_synthesis_content, history_md

            if "adversarial_analysis" in state:
                progress(
                    min(0.99, base_progress + (turn_weight * 0.6)),
                    desc=f"Cycle {active_turn}: Adversarial analysis complete.",
                )

                critiques = state["adversarial_analysis"].get("critiques_collection")
                if critiques and critiques.critiques:
                    critiques_dict = critiques.model_dump()
                else:
                    critiques_dict = {"critiques": []}

                yield initial_proposal_content, critiques_dict, final_synthesis_content, history_md

            if "synthesize_sequential" in state or "synthesize_architect" in state:
                progress(
                    min(0.99, base_progress + (turn_weight * 0.9)),
                    desc=f"Cycle {active_turn}: Synthesis complete.",
                )
                synthesis_output = state.get("synthesize_sequential") or state.get("synthesize_architect")

                final_synthesis_content = synthesis_output["current_artifact"].content
                history = synthesis_output["current_artifact"].history
                usage = synthesis_output.get("token_usage")
                usage_line = (
                    f"\n\n---\n**Cumulative Token Usage:** {usage.total_tokens} tokens "
                    f"({usage.input_tokens} prompt / {usage.output_tokens} completion across {usage.llm_calls} LLM calls)"
                    if usage
                    else ""
                )
                history_md = "\n".join([f"- {entry}" for entry in history]) + usage_line

                yield initial_proposal_content, critiques_dict, final_synthesis_content, history_md

        progress(1.0, desc=f"Workflow Finished after {max_turns_input} cycle(s)!")

    except gr.Error:
        raise
    except Exception as e:
        logger.exception(f"Gradio execution error: {e}")
        raise gr.Error(f"Execution failed: {e}")


def create_persona_ui(persona_name, is_adversary=False):
    available_personas = {**ADVERSARY_PERSONAS, **load_personas()}
    with gr.Accordion(f"{persona_name} Configuration", open=True):
        persona_selector = None
        if is_adversary:
            persona_selector = gr.Dropdown(
                label="Persona Type",
                choices=["NONE"] + list(available_personas.keys()),
                value="NONE",
            )

        provider = gr.Dropdown(
            label="LLM Provider",
            choices=["OpenAI", "Anthropic", "Google", "Ollama"],
            value=settings.default_provider,
        )

        with gr.Row():
            model = gr.Dropdown(
                label="Model Name",
                choices=FALLBACK_MODELS.get(settings.default_provider, []),
                value=settings.default_model,
                allow_custom_value=True,
                scale=4,
            )
            refresh_btn = gr.Button("🔄 Fetch Latest", scale=1)

        api_key = gr.Textbox(
            label="API Key",
            type="password",
            placeholder="Leave blank for env vars (Not needed for Ollama)",
        )
        base_url = gr.Textbox(
            label="Base URL",
            placeholder="Optional: http://localhost:11434 for Ollama or allowlisted endpoints",
            visible=settings.default_provider in ["Ollama", "OpenAI"],
        )
        temp = gr.Slider(
            label="Temperature",
            minimum=0.0,
            maximum=1.5,
            step=0.1,
            value=settings.default_temperature,
        )
        max_tokens = gr.Slider(
            label="Max Tokens",
            minimum=500,
            maximum=16000,
            step=500,
            value=settings.default_max_tokens,
        )

        def update_provider_settings(prov):
            choices = FALLBACK_MODELS.get(prov, [])
            show_base_url = prov in ["Ollama", "OpenAI"]
            return (
                gr.update(choices=choices, value=choices[0] if choices else None),
                gr.update(visible=show_base_url),
            )

        provider.change(fn=update_provider_settings, inputs=provider, outputs=[model, base_url])

        async def trigger_model_fetch(prov, key, url):
            new_models = await fetch_available_models(prov, key, url)
            return gr.update(choices=new_models, value=new_models[0] if new_models else None)

        refresh_btn.click(fn=trigger_model_fetch, inputs=[provider, api_key, base_url], outputs=[model])

        if is_adversary:
            def toggle_adversary_inputs(persona_type):
                disabled = persona_type == "NONE"
                return [gr.update(interactive=not disabled)] * 6

            persona_selector.change(
                fn=toggle_adversary_inputs,
                inputs=persona_selector,
                outputs=[provider, api_key, base_url, model, temp, max_tokens],
            )
            return persona_selector, provider, api_key, base_url, model, temp, max_tokens
        else:
            return provider, api_key, base_url, model, temp, max_tokens


# FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: the PG pool initializes dynamically when engine.py needs it
    yield
    # Shutdown: Safely drain and close the connection pool to prevent zombie connections
    await close_pg_pool()


fastapi_app = FastAPI(title="MASC Enterprise Studio UI", lifespan=lifespan)

# --- Gradio UI Layout ---
with gr.Blocks(theme=gr.themes.Soft(), title="MASC Enterprise Studio") as app:
    gr.Markdown("# MASC: Advanced Workflow Studio")
    all_ui_inputs = []

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 1. Task & Optional Initial Artifact")
            task_description_input = gr.Textbox(label="High-Level Goal / Audit Criteria", lines=4)
            all_ui_inputs.append(task_description_input)

            initial_artifact_input = gr.Textbox(
                label="Existing Artifact to Audit (Optional - Leave blank to generate from scratch)",
                placeholder="Paste existing code, architecture proposal, or policy document here to skip the Proposer step...",
                lines=4,
            )
            all_ui_inputs.append(initial_artifact_input)

            gr.Markdown("## 2. Workflow Configuration")
            with gr.Tabs():
                with gr.TabItem("Core Roles & Settings"):
                    all_ui_inputs.extend(create_persona_ui("Proposer"))
                    all_ui_inputs.extend(create_persona_ui("Synthesizer"))

                    synthesis_protocol_selector = gr.Radio(
                        label="Synthesis Protocol",
                        choices=["Sequential Refinement", "Architect"],
                        value="Sequential Refinement",
                    )
                    all_ui_inputs.append(synthesis_protocol_selector)

                    max_turns_input = gr.Slider(
                        label="Max Critique Cycles",
                        minimum=1,
                        maximum=settings.max_turns_limit,
                        step=1,
                        value=1,
                        info="Number of times the artifact loops back through adversarial critique.",
                    )
                    all_ui_inputs.append(max_turns_input)

                    stop_severity_selector = gr.Dropdown(
                        label="Early Convergence Threshold (Stop when no issues >= severity)",
                        choices=["CRITICAL", "HIGH", "MEDIUM", "LOW", "NONE"],
                        value="LOW",
                    )
                    all_ui_inputs.append(stop_severity_selector)

                    batch_constructive_checkbox = gr.Checkbox(
                        label="Batch Constructive Critiques in Sequential Mode",
                        value=False,
                        info="Applies all constructive fixes in a single pass per turn to reduce latency and token usage.",
                    )
                    all_ui_inputs.append(batch_constructive_checkbox)

                with gr.TabItem("Adversaries"):
                    for i in range(settings.max_adversaries):
                        all_ui_inputs.extend(create_persona_ui(f"Adversary Slot {i + 1}", is_adversary=True))

            run_button = gr.Button("Execute Graph", variant="primary")

        with gr.Column(scale=2):
            gr.Markdown("## 3. Telemetry & Results")
            with gr.Tabs():
                with gr.TabItem("Current Synthesis"):
                    final_synthesis_output = gr.Markdown(label="Hardened Artifact")
                with gr.TabItem("Process Breakdown"):
                    v1_proposal_output = gr.Markdown("### Initial Proposal / Input Artifact")
                    critiques_output = gr.JSON(label="Latest Adversarial Critiques")
                with gr.TabItem("Refinement History & Token Log"):
                    history_output = gr.Markdown()

    run_button.click(
        fn=run_masc_t_cycle,
        inputs=all_ui_inputs,
        outputs=[v1_proposal_output, critiques_output, final_synthesis_output, history_output],
    )

fastapi_app = gr.mount_gradio_app(
    fastapi_app,
    app,
    path="/",
    auth=verify_gradio_auth if settings.masc_api_key else None,
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.entrypoints.gradio_ui:fastapi_app", host="0.0.0.0", port=7860)
