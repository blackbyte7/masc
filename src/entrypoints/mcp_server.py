import logging
import uuid
from mcp.server.fastmcp import FastMCP

from src.core.engine import execute_masc_workflow
from src.core.state import MASCConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

mcp = FastMCP("MASC-Dialectics")


@mcp.tool()
async def trigger_masc_audit(task_description: str, config_json: str) -> str:
    """
    Executes a multi-agent adversarial audit on a given task or proposal over configured cycles.

    Args:
        task_description: The problem statement or proposal to be audited.
        config_json: A JSON string representing the MASCConfig schema. Must include 'max_turns' for cycle count.
    """
    try:
        config = MASCConfig.model_validate_json(config_json)
        thread_id = str(uuid.uuid4())

        final_state = None

        async for state in execute_masc_workflow(task_description, config, thread_id):
            final_state = state

        if not final_state:
            return "Error: Workflow failed to produce any output."

        # Extract the final synthesized artifact from whatever the last node was
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


if __name__ == "__main__":
    mcp.run()
