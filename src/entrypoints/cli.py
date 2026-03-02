import argparse
import asyncio
import logging
import sys
import uuid

from src.config.settings import settings
from src.core.engine import execute_masc_workflow
from src.core.state import MASCConfig, PersonaConfig, LLMConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def generate_template(filepath: str):
    """Generates a boilerplate JSON configuration file for the user."""
    default_llm = LLMConfig(provider=settings.default_provider, api_key="env", model_name=settings.default_model)
    template = MASCConfig(
        proposer=PersonaConfig(persona_name="Proposer", llm_config=default_llm),
        synthesizer=PersonaConfig(persona_name="Synthesizer", llm_config=default_llm),
        adversaries=[PersonaConfig(persona_name="DA", llm_config=default_llm)],
        synthesis_protocol="Sequential Refinement",
        max_turns=2  # Added max_turns to the template
    )

    with open(filepath, 'w') as f:
        f.write(template.model_dump_json(indent=4))
    print(f"✅ Configuration template generated at: {filepath}")


async def run_cli(task: str, config_path: str):
    print("\n" + "=" * 50)
    print("🧠 MASC Enterprise CLI Execution")
    print("=" * 50)

    try:
        with open(config_path, 'r') as f:
            config = MASCConfig.model_validate_json(f.read())
    except Exception as e:
        print(f"❌ Failed to parse config file '{config_path}': {e}")
        sys.exit(1)

    thread_id = str(uuid.uuid4())
    print(f"Task: {task}")
    print(f"Thread ID: {thread_id}")
    print(f"Synthesis Protocol: {config.synthesis_protocol}")
    print(f"Max Cycles: {config.max_turns}")
    print("-" * 50)

    try:
        final_content = ""
        async for state in execute_masc_workflow(task, config, thread_id):
            step_name = list(state.keys())[0]
            node_state = state[step_name]

            # Extract current turn if available
            current_turn = node_state.get("current_turn", "?")

            print(f"🔄 [Turn {current_turn}] Executed Node: [ {step_name.upper()} ]")

            if step_name in ["synthesize_sequential", "synthesize_architect"]:
                final_content = node_state["current_artifact"].content

        print("\n" + "=" * 50)
        print("🎯 FINAL SYNTHESIS ARTIFACT")
        print("=" * 50)
        print(final_content)
        print("\n✅ Workflow Completed Successfully.")

    except Exception as e:
        print(f"\n❌ Execution Error: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="MASC General-Purpose Dialectical AI")
    parser.add_argument("--task", type=str, help="The prompt or goal for the MASC workflow.")
    parser.add_argument("--config", type=str, help="Path to the JSON configuration file.")
    parser.add_argument("--generate-template", type=str, metavar="FILEPATH",
                        help="Generate a base JSON configuration file.")

    args = parser.parse_args()

    if args.generate_template:
        generate_template(args.generate_template)
        sys.exit(0)

    if not args.task or not args.config:
        parser.print_help()
        print("\nError: Both --task and --config are required to run an execution.")
        sys.exit(1)

    asyncio.run(run_cli(args.task, args.config))


if __name__ == "__main__":
    main()
