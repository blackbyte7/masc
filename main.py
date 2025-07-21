import gradio as gr
import os
import traceback
from dotenv import load_dotenv
from masc_engine import create_masc_t_graph, MASCConfig, PersonaConfig, LLMConfig, ADVERSARY_PERSONAS

load_dotenv()

MODEL_ZOO = {
    "OpenAI": ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
    "Anthropic": ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
    "Google": ["gemini-1.5-pro-latest", "gemini-1.0-pro", "gemini-1.5-flash-latest"],
}
MAX_ADVERSARIES = 4


def run_masc_t_cycle(task_desc, *config_inputs, progress=gr.Progress(track_tqdm=True)):
    """
    Orchestrates the MASC process as a generator, yielding real-time updates to the Gradio UI.
    """
    try:
        progress(0, desc="Parsing workflow configuration...")

        proposer_llm = LLMConfig(
            provider=config_inputs[0],
            api_key=config_inputs[1] or os.getenv(f"{config_inputs[0].upper()}_API_KEY"),
            model_name=config_inputs[2],
            temperature=config_inputs[3],
        )
        proposer_cfg = PersonaConfig(persona_name="Proposer", llm_config=proposer_llm)

        synth_llm = LLMConfig(
            provider=config_inputs[4],
            api_key=config_inputs[5] or os.getenv(f"{config_inputs[4].upper()}_API_KEY"),
            model_name=config_inputs[6],
            temperature=config_inputs[7],
        )
        synthesizer_cfg = PersonaConfig(persona_name="Synthesizer", llm_config=synth_llm)

        synthesis_protocol_input = config_inputs[8]

        adversary_cfgs = []
        adv_inputs = config_inputs[9:]
        for i in range(MAX_ADVERSARIES):
            chunk = adv_inputs[i * 5: (i + 1) * 5]
            if chunk[0] != "NONE":
                provider = chunk[1]
                api_key = chunk[2] or os.getenv(f"{provider.upper()}_API_KEY")
                if not api_key:
                    raise gr.Error(f"API Key for {provider} (Adversary {i + 1}) is missing.")
                adv_llm_cfg = LLMConfig(
                    provider=provider, api_key=api_key, model_name=chunk[3], temperature=chunk[4]
                )
                adversary_cfgs.append(PersonaConfig(persona_name=chunk[0], llm_config=adv_llm_cfg))

        if not adversary_cfgs:
            raise gr.Error("At least one Adversary must be configured.")

        masc_config = MASCConfig(
            proposer=proposer_cfg,
            synthesizer=synthesizer_cfg,
            adversaries=adversary_cfgs,
            synthesis_protocol=synthesis_protocol_input
        )

        progress(0.1, desc="Compiling MASC Graph...")
        graph = create_masc_t_graph()

        initial_state = {"task_description": task_desc, "config": masc_config}

        v1_proposal_content, critiques_json, final_synthesis_content, history_md = "", "", "", ""

        for state in graph.stream(initial_state, {"recursion_limit": 10}):
            if "propose" in state:
                progress(0.3, desc="Stage 1: Proposal received.")
                v1_proposal_content = state["propose"]["v1_proposal"].content
                yield v1_proposal_content, critiques_json, final_synthesis_content, history_md

            if "adversarial_analysis" in state:
                progress(0.6, desc="Stage 2: Adversarial analysis complete.")
                critiques_json = state["adversarial_analysis"]["critiques_collection"].json(indent=2)
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

        provider = gr.Dropdown(label="LLM Provider", choices=list(MODEL_ZOO.keys()), value="OpenAI")
        model = gr.Dropdown(label="Model Name", choices=MODEL_ZOO[provider.value], value=MODEL_ZOO[provider.value][0])
        api_key = gr.Textbox(label="API Key", type="password", placeholder="Leave blank to use .env")
        temp = gr.Slider(label="Temperature", minimum=0.0, maximum=1.5, step=0.1, value=0.1)

        def update_models(prov):
            choices = MODEL_ZOO.get(prov, [])
            return gr.update(choices=choices, value=choices[0] if choices else None)

        provider.change(fn=update_models, inputs=provider, outputs=model)

        if is_adversary:
            def toggle_adversary_inputs(persona_type):
                disabled = persona_type == "NONE"
                return (
                    gr.update(disabled=disabled),
                    gr.update(disabled=disabled),
                    gr.update(disabled=disabled),
                    gr.update(disabled=disabled),
                )

            persona_selector.change(
                fn=toggle_adversary_inputs,
                inputs=persona_selector,
                outputs=[provider, api_key, model, temp],
            )
            return persona_selector, provider, api_key, model, temp
        else:
            return provider, api_key, model, temp


with gr.Blocks(theme=gr.themes.Soft(), title="MASC Advanced Agent") as app:
    gr.Markdown("# MASC: Advanced Workflow Designer")
    all_ui_inputs = []
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 1. Task Description")
            task_description_input = gr.Textbox(label="High-Level Goal", lines=5)
            all_ui_inputs.append(task_description_input)

            gr.Markdown("## 2. Workflow Configuration")
            with gr.Tabs():
                with gr.TabItem("Core Roles"):
                    proposer_inputs = create_persona_ui("Proposer")
                    all_ui_inputs.extend(proposer_inputs)

                    synthesizer_inputs = create_persona_ui("Synthesizer")
                    all_ui_inputs.extend(synthesizer_inputs)

                    # NEW: UI for selecting synthesis protocol
                    synthesis_protocol_selector = gr.Radio(
                        label="Synthesis Protocol",
                        choices=['Sequential Refinement', 'Architect'],
                        value='Sequential Refinement',
                        info="Choose the method for resolving critiques. 'Architect' is a placeholder."
                    )
                    all_ui_inputs.append(synthesis_protocol_selector)

                with gr.TabItem("Adversaries"):
                    for i in range(MAX_ADVERSARIES):
                        all_ui_inputs.extend(create_persona_ui(f"Adversary Slot {i + 1}", is_adversary=True))

            run_button = gr.Button("Run MASC Workflow", variant="primary")

        with gr.Column(scale=2):
            gr.Markdown("## 3. Results")
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
    app.launch(debug=True)
