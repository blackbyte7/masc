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
    Executes a multi-agent adversarial audit on a given task or proposal.

    Args:
        task_description: The problem statement or proposal to be audited.
        config_json: A JSON string representing the MASCConfig schema.
    """
    try:
        config = MASCConfig.model_validate_json(config_json)
        thread_id = str(uuid.uuid4())

        final_artifact = None

        async for state in execute_masc_workflow(task_description, config, thread_id):
            if "synthesize_sequential" in state:
                final_artifact = state["synthesize_sequential"]["final_synthesis"].content
            elif "synthesize_architect" in state:
                final_artifact = state["synthesize_architect"]["final_synthesis"].content

        if not final_artifact:
            return "Error: Workflow failed to produce a final synthesis."

        return f"MASC Dialectical Audit Complete. Final Hardened Output:\n\n{final_artifact}"

    except Exception as e:
        return f"MASC Execution Error: {str(e)}"


if __name__ == "__main__":
    mcp.run()
