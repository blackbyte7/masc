import importlib
import inspect
import json
import logging
import sys
import uuid
from typing import Any, Optional

from src.config.settings import settings, validate_base_url
from src.core.engine import close_pg_pool, execute_masc_workflow, resume_masc_workflow
from src.core.personas import ADVERSARY_PERSONAS, load_personas, resolve_adversary_persona
from src.core.state import CritiquesCollection, MASCConfig

# Strictly log to stderr so MCP stdio JSON-RPC messages on stdout are never corrupted
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class _FallbackMCP:
    """Lightweight fallback when neither 'mcp' nor 'fastmcp' is installed in a test environment."""

    def __init__(self, name: str) -> None:
        self.name = name

    @staticmethod
    def tool():
        def _decorator(func: Any) -> Any:
            return func

        return _decorator

    @staticmethod
    def run() -> None:
        raise RuntimeError("Neither 'mcp' nor 'fastmcp' package is installed. Run: pip install mcp fastmcp")


def _create_mcp_server(server_name: str) -> Any:
    """Initializes FastMCP dynamically to avoid IDE stub resolution conflicts across mcp/fastmcp versions."""
    for module_name in ("mcp.server.fastmcp", "fastmcp"):
        try:
            mod = importlib.import_module(module_name)
            fastmcp_cls: Any = getattr(mod, "FastMCP", None)
            if callable(fastmcp_cls):
                return fastmcp_cls(server_name)
        except ImportError:
            continue
    return _FallbackMCP(server_name)


def _register_mcp_tool(server: Any, func: Any) -> None:
    """Registers an async function with FastMCP while keeping the function directly awaitable for unit tests."""
    tool_decorator = getattr(server, "tool", None)
    if callable(tool_decorator):
        tool_decorator()(func)


mcp = _create_mcp_server("MASC-Dialectics")


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
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if "initial_artifact" in sig.parameters or has_var_kw:
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
        if last_node_name == "__interrupt__":
            interrupt_data = final_state["__interrupt__"]
            critiques_obj = interrupt_data.get("critiques_collection")
            critiques_str = (
                critiques_obj.model_dump_json(indent=2) if hasattr(critiques_obj, "model_dump_json") else "{}"
            )
            return (
                f"MASC Audit Paused for Human Approval (thread_id: {thread_id}).\n\n"
                f"### Pending Critiques:\n{critiques_str}"
            )

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
        override = (
            CritiquesCollection.model_validate_json(critiques_override_json)
            if critiques_override_json
            else None
        )
        final_state = None
        async for state in resume_masc_workflow(
            thread_id=thread_id,
            critiques_override=override,
            approve=approve,
        ):
            final_state = state

        if not final_state:
            return "Error: Resumed workflow produced no output."

        last_node_name = list(final_state.keys())[0]
        if last_node_name == "aborted":
            return f"MASC Workflow on thread '{thread_id}' was aborted by reviewer."

        final_artifact = final_state[last_node_name].get("current_artifact")
        if not final_artifact:
            return f"MASC Workflow resumed on '{thread_id}', state: {json.dumps(final_state, default=str)}"

        history_formatted = "\n".join([f"- {h}" for h in final_artifact.history])
        return (
            f"MASC Resumed Audit Complete (thread_id: {thread_id}).\n\n"
            f"### Refinement History:\n{history_formatted}\n\n"
            f"### Final Hardened Output:\n\n{final_artifact.content}"
        )
    except Exception as e:
        return f"MASC Resume Error: {str(e)}"
    finally:
        await close_pg_pool()


_register_mcp_tool(mcp, trigger_masc_audit)
_register_mcp_tool(mcp, resume_masc_audit)


if __name__ == "__main__":
    mcp.run()
