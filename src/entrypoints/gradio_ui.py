import logging
import traceback
import uuid

import gradio as gr

from src.config.settings import settings
from src.core.engine import execute_masc_workflow
from src.core.personas import ADVERSARY_PERSONAS
from src.core.state import MASCConfig, PersonaConfig, LLMConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_ZOO = {
    "OpenAI": ["gpt-4.1", "gpt-4.1-mini", "o3", "o4-mini", "gpt-4o"],
    "Anthropic": ["claude-4-opus-20250514", "claude-4-sonnet-20250514", "claude-3-7-sonnet-20250219"],
    "Google": ["gemini-3.1-pro-preview", "gemini-3-flash-preview", "gemini-2.5-pro", "gemini-2.5-flash",
               "gemini-2.5-flash-lite"],
    "Ollama": ["llama3", "mistral", "qwen2", "phi3"]
}


async def run_masc_t_cycle(task_desc, *config_inputs, progress=gr.Progress(track_tqdm=True)):
    """Orchestrates the MASC process as an async generator, updating Gradio UI."""
    try:
        progress(0, desc="Parsing workflow configuration...")

        # 1. Unpack Proposer (Index 0 to 5)
        proposer_llm = LLMConfig(
            provider=config_inputs[0],
            api_key=config_inputs[1] or "",
            base_url=config_inputs[2] or None,
            model_name=config_inputs[3],
            temperature=config_inputs[4],
            max_tokens=config_inputs[5]
        )
        proposer_cfg = PersonaConfig(persona_name="Proposer", llm_config=proposer_llm)

        # 2. Unpack Synthesizer (Index 6 to 11)
        synth_llm = LLMConfig(
            provider=config_inputs[6],
            api_key=config_inputs[7] or "",
            base_url=config_inputs[8] or None,
            model_name=config_inputs[9],
            temperature=config_inputs[10],
            max_tokens=config_inputs[11]
        )
        synthesizer_cfg = PersonaConfig(persona_name="Synthesizer", llm_config=synth_llm)

        synthesis_protocol_input = config_inputs[12]
        max_turns_input = int(config_inputs[13])

        # 3. Unpack Adversaries dynamically (Chunks of 7 starting at index 14)
        adversary_cfgs = []
        adv_inputs = config_inputs[14:]

        for i in range(settings.max_adversaries):
            chunk = adv_inputs[i * 7: (i + 1) * 7]
            if chunk[0] != "NONE":
                adv_llm_cfg = LLMConfig(
                    provider=chunk[1],
                    api_key=chunk[2] or "",
                    base_url=chunk[3] or None,
                    model_name=chunk[4],
                    temperature=chunk[5],
                    max_tokens=chunk[6]
                )
                adversary_cfgs.append(PersonaConfig(persona_name=chunk[0], llm_config=adv_llm_cfg))

        if not adversary_cfgs:
            raise gr.Error("At least one Adversary must be configured.")

        masc_config = MASCConfig(
            proposer=proposer_cfg,
            synthesizer=synthesizer_cfg,
            adversaries=adversary_cfgs,
            synthesis_protocol=synthesis_protocol_input,
            max_turns=max_turns_input
        )

        progress(0.05, desc="Compiling MASC Graph & Connecting to Telemetry...")
        thread_id = str(uuid.uuid4())

        # Initialize empty states for the UI
        initial_proposal_content = ""
        critiques_dict = {"status": "Awaiting adversarial analysis..."}
        final_synthesis_content = ""
        history_md = "*Workflow initialized...*"

        async for state in execute_masc_workflow(task_desc, masc_config, thread_id):

            node_name = list(state.keys())[0]
            current_turn = state[node_name].get("current_turn", 1)

            base_progress = (current_turn - 1) / max_turns_input
            turn_weight = 1.0 / max_turns_input

            if "propose" in state:
                progress(base_progress + (turn_weight * 0.2),
                         desc=f"Cycle {current_turn}: Generating Initial Proposal...")

                initial_proposal_content = state["propose"]["current_artifact"].content
                history = state["propose"]["current_artifact"].history
                history_md = "\n".join([f"- {entry}" for entry in history])

                yield initial_proposal_content, critiques_dict, final_synthesis_content, history_md

            if "adversarial_analysis" in state:
                progress(base_progress + (turn_weight * 0.6),
                         desc=f"Cycle {current_turn}: Adversarial analysis complete.")

                critiques = state["adversarial_analysis"].get("critiques_collection")
                if critiques and critiques.critiques:
                    critiques_dict = critiques.model_dump()
                else:
                    critiques_dict = {"critiques": []}

                yield initial_proposal_content, critiques_dict, final_synthesis_content, history_md

            if "synthesize_sequential" in state or "synthesize_architect" in state:
                progress(base_progress + (turn_weight * 0.9), desc=f"Cycle {current_turn}: Synthesis complete.")
                synthesis_output = state.get("synthesize_sequential") or state.get("synthesize_architect")

                final_synthesis_content = synthesis_output["current_artifact"].content
                history = synthesis_output["current_artifact"].history
                history_md = "\n".join([f"- {entry}" for entry in history])

                yield initial_proposal_content, critiques_dict, final_synthesis_content, history_md

        progress(1.0, desc=f"Workflow Finished after {max_turns_input} cycles!")

    except Exception as e:
        error_msg = f"An error occurred: {e}\n\n{traceback.format_exc()}"
        raise gr.Error(error_msg)


def create_persona_ui(persona_name, is_adversary=False):
    with gr.Accordion(f"{persona_name} Configuration", open=True):
        persona_selector = None
        if is_adversary:
            persona_selector = gr.Dropdown(
                label="Persona Type", choices=["NONE"] + list(ADVERSARY_PERSONAS.keys()), value="NONE"
            )

        provider = gr.Dropdown(label="LLM Provider", choices=list(MODEL_ZOO.keys()), value=settings.default_provider)
        # allow_custom_value=True guarantees future-proofing
        model = gr.Dropdown(label="Model Name", choices=MODEL_ZOO[settings.default_provider],
                            value=settings.default_model, allow_custom_value=True)
        api_key = gr.Textbox(label="API Key", type="password",
                             placeholder="Leave blank for env vars (Not needed for Ollama)")
        base_url = gr.Textbox(label="Base URL",
                              placeholder="Optional: http://localhost:11434 for Ollama or custom endpoints",
                              visible=False)
        temp = gr.Slider(label="Temperature", minimum=0.0, maximum=1.5, step=0.1, value=settings.default_temperature)
        max_tokens = gr.Slider(label="Max Tokens", minimum=500, maximum=16000, step=500,
                               value=settings.default_max_tokens)

        def update_provider_settings(prov):
            choices = MODEL_ZOO.get(prov, [])
            show_base_url = prov in ["Ollama", "OpenAI"]
            return (
                gr.update(choices=choices, value=choices[0] if choices else None),
                gr.update(visible=show_base_url)
            )

        provider.change(fn=update_provider_settings, inputs=provider, outputs=[model, base_url])

        if is_adversary:
            def toggle_adversary_inputs(persona_type):
                disabled = persona_type == "NONE"
                return [gr.update(disabled=disabled)] * 6

            persona_selector.change(
                fn=toggle_adversary_inputs,
                inputs=persona_selector,
                outputs=[provider, api_key, base_url, model, temp, max_tokens],
            )
            return persona_selector, provider, api_key, base_url, model, temp, max_tokens
        else:
            return provider, api_key, base_url, model, temp, max_tokens


with gr.Blocks(theme=gr.themes.Soft(), title="MASC Enterprise Studio") as app:
    gr.Markdown("# MASC: Advanced Workflow Studio")
    all_ui_inputs = []

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 1. Task Description")
            task_description_input = gr.Textbox(label="High-Level Goal", lines=5)
            all_ui_inputs.append(task_description_input)

            gr.Markdown("## 2. Workflow Configuration")
            with gr.Tabs():
                with gr.TabItem("Core Roles & Settings"):
                    all_ui_inputs.extend(create_persona_ui("Proposer"))
                    all_ui_inputs.extend(create_persona_ui("Synthesizer"))

                    synthesis_protocol_selector = gr.Radio(
                        label="Synthesis Protocol",
                        choices=['Sequential Refinement', 'Architect'],
                        value='Sequential Refinement'
                    )
                    all_ui_inputs.append(synthesis_protocol_selector)

                    max_turns_input = gr.Slider(
                        label="Max Critique Cycles",
                        minimum=1,
                        maximum=10,
                        step=1,
                        value=1,
                        info="Number of times the artifact loops back through adversarial critique."
                    )
                    all_ui_inputs.append(max_turns_input)

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
                    v1_proposal_output = gr.Markdown("### Initial Proposal (Cycle 1)")
                    # Replaced gr.Code with gr.JSON for better object inspection
                    critiques_output = gr.JSON(label="Latest Adversarial Critiques")
                with gr.TabItem("Refinement History Log"):
                    history_output = gr.Markdown()

    run_button.click(
        fn=run_masc_t_cycle,
        inputs=all_ui_inputs,
        outputs=[v1_proposal_output, critiques_output, final_synthesis_output, history_output],
    )

if __name__ == "__main__":
    app.launch()
