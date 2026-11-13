import importlib
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

from src.core.engine import close_pg_pool, execute_masc_workflow, resume_masc_workflow
from src.core.state import CritiquesCollection, MASCConfig
from src.core.validation import ConfigValidationError, validate_masc_config, validate_task_inputs

logger = logging.getLogger(__name__)


class _FallbackMCP:
    """Lightweight fallback when neither 'mcp' nor 'fastmcp' is installed in a test environment."""

    def __init__(self, name: str, **_kwargs: Any) -> None:
        self.name = name

    @staticmethod
    def tool():
        def _decorator(func: Any) -> Any:
            return func

        return _decorator

    @staticmethod
    def run() -> None:
        raise RuntimeError("The 'mcp' package is not installed. Run: pip install mcp")


@asynccontextmanager
async def _server_lifespan(_server: Any) -> AsyncIterator[None]:
    """Keeps the shared PostgreSQL pool open for the server's lifetime and drains it once on shutdown."""
    try:
        yield
    finally:
        await close_pg_pool()


def _create_mcp_server(server_name: str) -> Any:
    """Initializes FastMCP dynamically to avoid IDE stub resolution conflicts across mcp/fastmcp versions."""
    for module_name in ("mcp.server.fastmcp", "fastmcp"):
        try:
            mod = importlib.import_module(module_name)
        except ImportError:
            continue
        fastmcp_cls: Any = getattr(mod, "FastMCP", None)
        if not callable(fastmcp_cls):
            continue
        try:
            return fastmcp_cls(server_name, lifespan=_server_lifespan)
        except TypeError:
            # Older FastMCP releases do not accept a lifespan hook.
            return fastmcp_cls(server_name)
    return _FallbackMCP(server_name)


def _register_mcp_tool(server: Any, func: Any) -> None:
    """Registers an async function with FastMCP while keeping the function directly awaitable for unit tests."""
    tool_decorator = getattr(server, "tool", None)
    if callable(tool_decorator):
        tool_decorator()(func)


mcp = _create_mcp_server("MASC-Dialectics")


def _format_interrupt(thread_id: str, interrupt_data: dict[str, Any], resumed: bool = False) -> str:
    critiques_obj = interrupt_data.get("critiques_collection")
    critiques_str = critiques_obj.model_dump_json(indent=2) if hasattr(critiques_obj, "model_dump_json") else "{}"
    prefix = "MASC Resumed Audit Paused Again" if resumed else "MASC Audit Paused"
    return (
        f"{prefix} for Human Approval (thread_id: {thread_id}, turn: {interrupt_data.get('current_turn', 1)}).\n"
        "Call resume_masc_audit to approve, edit or reject the pending critiques.\n\n"
        f"### Pending Critiques:\n{critiques_str}"
    )


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
        try:
            validate_task_inputs(task_description, initial_artifact)
            config = MASCConfig.model_validate_json(config_json)
            validate_masc_config(config)
        except (ConfigValidationError, ValueError) as exc:
            return f"MASC Validation Error: {exc}"

        thread_id = str(uuid.uuid4())
        final_state = None

        async for state in execute_masc_workflow(
            task_description, config, thread_id, initial_artifact=initial_artifact
        ):
            final_state = state

        if not final_state:
            return "Error: Workflow failed to produce any output."

        last_node_name = next(iter(final_state))
        if last_node_name == "__interrupt__":
            return _format_interrupt(thread_id, final_state["__interrupt__"])

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
        logger.exception("MCP trigger_masc_audit failed")
        return f"MASC Execution Error: {str(e)}"


async def resume_masc_audit(
    thread_id: str,
    approve: bool = True,
    critiques_override_json: Optional[str] = None,
) -> str:
    """
    Resumes a paused Human-in-the-Loop MASC workflow by thread_id.

    Args:
        thread_id: The thread ID of the paused MASC execution.
        approve: Whether to approve and proceed with synthesis (False aborts the run).
        critiques_override_json: Optional JSON string of CritiquesCollection to replace pending critiques.
    """
    try:
        override = CritiquesCollection.model_validate_json(critiques_override_json) if critiques_override_json else None
        final_state = None
        async for state in resume_masc_workflow(
            thread_id=thread_id,
            critiques_override=override,
            approve=approve,
        ):
            final_state = state

        if not final_state:
            return f"Error: Resumed workflow on '{thread_id}' produced no output (is the thread paused?)."

        last_node_name = next(iter(final_state))
        if last_node_name == "aborted":
            return f"MASC Workflow on thread '{thread_id}' was aborted by reviewer."
        if last_node_name == "__interrupt__":
            return _format_interrupt(thread_id, final_state["__interrupt__"], resumed=True)

        final_artifact = final_state[last_node_name].get("current_artifact")
        if not final_artifact:
            return f"Error: Resumed workflow on '{thread_id}' completed without a hardened artifact."

        history_formatted = "\n".join([f"- {h}" for h in final_artifact.history])
        return (
            f"MASC Resumed Audit Complete (thread_id: {thread_id}).\n\n"
            f"### Refinement History:\n{history_formatted}\n\n"
            f"### Final Hardened Output:\n\n{final_artifact.content}"
        )
    except Exception as e:
        logger.exception("MCP resume_masc_audit failed")
        return f"MASC Resume Error: {str(e)}"


_register_mcp_tool(mcp, trigger_masc_audit)
_register_mcp_tool(mcp, resume_masc_audit)


def run() -> None:
    """Runs the MCP server over stdio with MASC logging and a psycopg-compatible event loop."""
    from src.utils.logging_config import configure_logging
    from src.utils.runtime import configure_event_loop_policy

    configure_logging()
    configure_event_loop_policy()
    mcp.run()


if __name__ == "__main__":
    run()
