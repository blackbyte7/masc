import inspect
import logging
import sys
import uuid
from typing import Optional

from mcp.server.fastmcp import FastMCP

from src.config.settings import settings, validate_base_url
from src.core.engine import close_pg_pool, execute_masc_workflow
from src.core.personas import ADVERSARY_PERSONAS, load_personas, resolve_adversary_persona
from src.core.state import MASCConfig

# Strictly log to stderr so MCP stdio JSON-RPC messages on stdout are never corrupted
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

mcp = FastMCP("MASC-Dialectics")


@mcp.tool()
async def trigger_masc_audit(
    task_description: str,
    config_json: str,
    initial_artifact: Optional[str] = None,
) -> str:
    """
    Executes a multi-agent adversarial audit on a given task or existing artifact over configured cycles.

    Args:
        task_description: The problem statement, requirements, or audit criteria.
        config_json: A JSON string representing the MASCConfig schema. Must include 'max_turns' for cycle count.
        initial_artifact: Optional existing code/text artifact to audit directly without running the Proposer node.
    """
    try:
        if not task_description or not task_description.strip():
            return "MASC Validation Error: task_description cannot be empty."
        if len(task_description) > settings.max_task_length:
            return f"MASC Validation Error: task_description exceeds max length ({settings.max_task_length})."

        config = MASCConfig.model_validate_json(config_json)
        if len(config.adversaries) > settings.max_adversaries:
            return f"MASC Validation Error: Exceeded max adversaries ({settings.max_adversaries})."
        if config.max_turns > settings.max_turns_limit:
            return f"MASC Validation Error: Exceeded max turns limit ({settings.max_turns_limit})."

        available_personas = {**ADVERSARY_PERSONAS, **load_personas()}
        invalid_personas = []
        for adv in config.adversaries:
            try:
                resolve_adversary_persona(adv, available_personas)
            except ValueError:
                invalid_personas.append(adv.persona_name)

        if invalid_personas:
            return f"MASC Validation Error: Unknown adversary persona(s): {invalid_personas}."

        for persona in [config.proposer, config.synthesizer, *config.adversaries]:
            validate_base_url(persona.llm_config.base_url, settings)

        thread_id = str(uuid.uuid4())
        final_state = None

        sig = inspect.signature(execute_masc_workflow)
        if "initial_artifact" in sig.parameters:
            workflow_gen = execute_masc_workflow(
                task_description, config, thread_id, initial_artifact=initial_artifact
            )
        else:
            workflow_gen = execute_masc_workflow(task_description, config, thread_id)

        async for state in workflow_gen:
            final_state = state

        if not final_state:
            return "Error: Workflow failed to produce any output."

        last_node_name = list(final_state.keys())[0]
        final_artifact = final_state[last_node_name].get("current_artifact")

        if not final_artifact:
            return "Error: Workflow completed but failed to yield a hardened artifact."

        history_formatted = "\n".join([f"- {h}" for h in final_artifact.history])

        return (
            f"MASC Dialectical Audit Complete ({config.max_turns} cycles).\n\n"
            f"### Refinement History:\n{history_formatted}\n\n"
            f"### Final Hardened Output:\n\n{final_artifact.content}"
        )

    except Exception as e:
        return f"MASC Execution Error: {str(e)}"
    finally:
        await close_pg_pool()


if __name__ == "__main__":
    mcp.run()
