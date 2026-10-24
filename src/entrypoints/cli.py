import argparse
import asyncio
import logging
import sys
import uuid
from typing import Optional

from src.config.settings import settings
from src.core.engine import close_pg_pool, execute_masc_workflow
from src.core.state import LLMConfig, MASCConfig, PersonaConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def generate_template(filepath: str):
    """Generates a boilerplate JSON configuration file for the user."""
    default_llm = LLMConfig(
        provider=settings.default_provider,
        api_key="",
        model_name=settings.default_model,
        temperature=settings.default_temperature,
        max_tokens=settings.default_max_tokens,
    )
    template = MASCConfig(
        proposer=PersonaConfig(persona_name="Proposer", llm_config=default_llm),
        synthesizer=PersonaConfig(persona_name="Synthesizer", llm_config=default_llm),
        adversaries=[PersonaConfig(persona_name="DA", llm_config=default_llm)],
        synthesis_protocol="Sequential Refinement",
        max_turns=2,
    )

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(template.model_dump_json(indent=4))
    print(f"✅ Configuration template generated at: {filepath}")


async def run_cli(
    task: str,
    config_path: str,
    artifact_file: Optional[str] = None,
    auto_personas: bool = False,
    verify_tools: bool = False,
):
    print("\n" + "=" * 50)
    print("🧠 MASC Enterprise CLI Execution")
    print("=" * 50)

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = MASCConfig.model_validate_json(f.read())
    except Exception as e:
        print(f"❌ Failed to parse config file '{config_path}': {e}")
        sys.exit(1)

    if auto_personas:
        config.auto_select_adversaries = True
    if verify_tools:
        config.enable_verification_tools = True

    initial_artifact: Optional[str] = None
    if artifact_file:
        try:
            with open(artifact_file, "r", encoding="utf-8") as af:
                initial_artifact = af.read()
        except Exception as e:
            print(f"❌ Failed to read initial artifact file '{artifact_file}': {e}")
            sys.exit(1)

    thread_id = str(uuid.uuid4())
    print(f"Task: {task}")
    print(f"Thread ID: {thread_id}")
    print(f"Synthesis Protocol: {config.synthesis_protocol}")
    print(f"Max Cycles: {config.max_turns}")
    if config.auto_select_adversaries:
        print("Auto-Persona Router: Enabled")
    if initial_artifact or config.initial_artifact:
        print("Mode: Audit-Only (Skipping Proposer)")
    print("-" * 50)

    try:
        final_content = ""
        final_usage = None
        async for state in execute_masc_workflow(
            task, config, thread_id, initial_artifact=initial_artifact
        ):
            step_name = list(state.keys())[0]
            node_state = state[step_name]

            raw_turn = node_state.get("current_turn", 1)
            display_turn = (
                raw_turn - 1
                if step_name in ("synthesize_sequential", "synthesize_architect") and raw_turn > 1
                else raw_turn
            )

            print(f"🔄 [Turn {display_turn}] Executed Node: [ {step_name.upper()} ]")

            if step_name in ("synthesize_sequential", "synthesize_architect"):
                final_content = node_state["current_artifact"].content
            if "token_usage" in node_state:
                final_usage = node_state["token_usage"]

        print("\n" + "=" * 50)
        print("🎯 FINAL SYNTHESIS ARTIFACT")
        print("=" * 50)
        print(final_content)
        if final_usage:
            print(
                f"\n📊 Token Usage: {final_usage.total_tokens} total "
                f"({final_usage.input_tokens} in / {final_usage.output_tokens} out across {final_usage.llm_calls} LLM calls)"
            )
        print("\n✅ Workflow Completed Successfully.")

    except Exception as e:
        print(f"\n❌ Execution Error: {e}")
        sys.exit(1)
    finally:
        await close_pg_pool()


def main():
    parser = argparse.ArgumentParser(description="MASC General-Purpose Dialectical AI")
    parser.add_argument("--task", type=str, help="The prompt or goal for the MASC workflow.")
    parser.add_argument("--config", type=str, help="Path to the JSON configuration file.")
    parser.add_argument(
        "--initial-artifact",
        type=str,
        metavar="ARTIFACT_FILEPATH",
        help="Optional path to an existing file to audit directly (skips Proposer).",
    )
    parser.add_argument(
        "--auto-personas",
        action="store_true",
        help="Optional: enable the Auto-Persona Router to dynamically select relevant critics.",
    )
    parser.add_argument(
        "--verify-tools",
        action="store_true",
        help="Optional: enable deterministic static verification checks before adversary critique.",
    )
    parser.add_argument(
        "--generate-template",
        type=str,
        metavar="FILEPATH",
        help="Generate a base JSON configuration file.",
    )

    args = parser.parse_args()

    if args.generate_template:
        generate_template(args.generate_template)
        sys.exit(0)

    if not args.task or not args.config:
        parser.print_help()
        print("\nError: Both --task and --config are required to run an execution.")
        sys.exit(1)

    asyncio.run(
        run_cli(
            args.task,
            args.config,
            artifact_file=args.initial_artifact,
            auto_personas=args.auto_personas,
            verify_tools=args.verify_tools,
        )
    )


if __name__ == "__main__":
    main()
