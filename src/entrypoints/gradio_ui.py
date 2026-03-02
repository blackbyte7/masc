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
    "OpenAI": ["gpt-4o", "o3-mini", "gpt-4.5-preview"],
    "Anthropic": ["claude-3-7-sonnet-20250219", "claude-4-opus-20250514", "claude-4-sonnet-20250514"],
    "Google": ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"],
}


async def run_masc_t_cycle(task_desc, *config_inputs, progress=gr.Progress(track_tqdm=True)):
    """Orchestrates the MASC process as an async generator, updating Gradio UI."""
    try:
        progress(0, desc="Parsing workflow configuration...")

        # 1. Unpack Proposer
        proposer_llm = LLMConfig(
            provider=config_inputs[0],
            api_key=config_inputs[1] or "",
            model_name=config_inputs[2],
            temperature=config_inputs[3],
            max_tokens=config_inputs[4]
        )
        proposer_cfg = PersonaConfig(persona_name="Proposer", llm_config=proposer_llm)

        # 2. Unpack Synthesizer
        synth_llm = LLMConfig(
            provider=config_inputs[5],
            api_key=config_inputs[6] or "",
            model_name=config_inputs[7],
            temperature=config_inputs[8],
            max_tokens=config_inputs[9]
        )
        synthesizer_cfg = PersonaConfig(persona_name="Synthesizer", llm_config=synth_llm)

        synthesis_protocol_input = config_inputs[10]

        # 3. Unpack Adversaries dynamically based on settings
        adversary_cfgs = []
        adv_inputs = config_inputs[11:]

        for i in range(settings.max_adversaries):
            chunk = adv_inputs[i * 6: (i + 1) * 6]
            if chunk[0] != "NONE":
                adv_llm_cfg = LLMConfig(
                    provider=chunk[1],
                    api_key=chunk[2] or "",
                    model_name=chunk[3],
                    temperature=chunk[4],
                    max_tokens=chunk[5]
                )
                adversary_cfgs.append(PersonaConfig(persona_name=chunk[0], llm_config=adv_llm_cfg))

        if not adversary_cfgs:
            raise gr.Error("At least one Adversary must be configured.")

        # 4. Construct Strict Config
        masc_config = MASCConfig(
            proposer=proposer_cfg,
            synthesizer=synthesizer_cfg,
            adversaries=adversary_cfgs,
            synthesis_protocol=synthesis_protocol_input
        )

        progress(0.1, desc="Compiling MASC Graph...")
        thread_id = str(uuid.uuid4())

        v1_proposal_content, critiques_json, final_synthesis_content, history_md = "", "", "", ""

        # 5. Execute Core Engine
        async for state in execute_masc_workflow(task_desc, masc_config, thread_id):
            if "propose" in state:
                progress(0.3, desc="Stage 1: Proposal received.")
                v1_proposal_content = state["propose"]["v1_proposal"].content
                yield v1_proposal_content, critiques_json, final_synthesis_content, history_md

            if "adversarial_analysis" in state:
                progress(0.6, desc="Stage 2: Adversarial analysis complete.")
                critiques = state["adversarial_analysis"].get("critiques_collection")
                if critiques and critiques.critiques:
                    critiques_json = critiques.model_dump_json(indent=2)
                else:
                    critiques_json = '{"critiques": []}'
                yield v1_proposal_content, critiques_json, final_synthesis_content, history_md

            synthesis_output = state.get("synthesize_sequential") or state.get("synthesize_architect")
            if synthesis_output:
                progress(0.9, desc="Stage 3: Synthesis complete.")
                final_synthesis_content = synthesis_output["final_synthesis"].content
                history = synthesis_output["final_synthesis"].history
                history_md = "\n".join([f"- {entry}" for entry in history])
                yield v1_proposal_content, critiques_json, final_synthesis_content, history_md

        progress(1.0, desc="Workflow Finished!")

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
        model = gr.Dropdown(label="Model Name", choices=MODEL_ZOO[settings.default_provider],
                            value=settings.default_model)
        api_key = gr.Textbox(label="API Key", type="password", placeholder="Leave blank to use environment variables")
        temp = gr.Slider(label="Temperature", minimum=0.0, maximum=1.5, step=0.1, value=settings.default_temperature)
        max_tokens = gr.Slider(label="Max Tokens", minimum=500, maximum=16000, step=500,
                               value=settings.default_max_tokens)

        def update_models(prov):
            choices = MODEL_ZOO.get(prov, [])
            return gr.update(choices=choices, value=choices[0] if choices else None)

        provider.change(fn=update_models, inputs=provider, outputs=model)

        if is_adversary:
            def toggle_adversary_inputs(persona_type):
                disabled = persona_type == "NONE"
                return (
                    gr.update(disabled=disabled), gr.update(disabled=disabled),
                    gr.update(disabled=disabled), gr.update(disabled=disabled),
                    gr.update(disabled=disabled)
                )

            persona_selector.change(
                fn=toggle_adversary_inputs,
                inputs=persona_selector,
                outputs=[provider, api_key, model, temp, max_tokens],
            )
            return persona_selector, provider, api_key, model, temp, max_tokens
        else:
            return provider, api_key, model, temp, max_tokens


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
                with gr.TabItem("Core Roles"):
                    all_ui_inputs.extend(create_persona_ui("Proposer"))
                    all_ui_inputs.extend(create_persona_ui("Synthesizer"))

                    synthesis_protocol_selector = gr.Radio(
                        label="Synthesis Protocol",
                        choices=['Sequential Refinement', 'Architect'],
                        value='Sequential Refinement'
                    )
                    all_ui_inputs.append(synthesis_protocol_selector)

                with gr.TabItem("Adversaries"):
                    for i in range(settings.max_adversaries):
                        all_ui_inputs.extend(create_persona_ui(f"Adversary Slot {i + 1}", is_adversary=True))

            run_button = gr.Button("Execute Graph", variant="primary")

        with gr.Column(scale=2):
            gr.Markdown("## 3. Telemetry & Results")
            with gr.Tabs():
                with gr.TabItem("Final Synthesis"):
                    final_synthesis_output = gr.Markdown()
                with gr.TabItem("Process Breakdown"):
                    v1_proposal_output = gr.Markdown("### v1 Proposal")
                    critiques_output = gr.Code(label="Adversarial Critiques", language="json")
                with gr.TabItem("Refinement History"):
                    history_output = gr.Markdown()

    run_button.click(
        fn=run_masc_t_cycle,
        inputs=all_ui_inputs,
        outputs=[v1_proposal_output, critiques_output, final_synthesis_output, history_output],
    )

if __name__ == "__main__":
    app.launch()
