import asyncio
import logging
import traceback
from collections import deque
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Literal, Optional

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from psycopg_pool import AsyncConnectionPool

from src.config.settings import settings, validate_base_url
from src.core.personas import ADVERSARY_PERSONAS, load_personas
from src.core.state import (
    Artifact,
    Critique,
    CritiqueItem,
    CritiqueLLMOutput,
    CritiquesCollection,
    MASCConfig,
    MASCState,
    PersonaConfig,
)

logger = logging.getLogger(__name__)

_pg_pool: Optional[AsyncConnectionPool] = None
_pg_pool_lock = asyncio.Lock()
_pg_setup_done: bool = False
_memory_saver = MemorySaver()
_memory_thread_order: deque[str] = deque()

SEVERITY_RANK = {
    "CRITICAL": 0,
    "HIGH": 1,
    "MEDIUM": 2,
    "LOW": 3,
}


def _track_and_prune_memory_threads(thread_id: str):
    """Bounds the singleton MemorySaver storage to prevent unbounded memory leaks."""
    if thread_id in _memory_thread_order:
        _memory_thread_order.remove(thread_id)
    _memory_thread_order.append(thread_id)

    while len(_memory_thread_order) > settings.max_in_memory_threads:
        evicted_thread = _memory_thread_order.popleft()
        if hasattr(_memory_saver, "storage"):
            _memory_saver.storage.pop(evicted_thread, None)
        if hasattr(_memory_saver, "writes"):
            keys_to_remove = [k for k in _memory_saver.writes if k and k[0] == evicted_thread]
            for k in keys_to_remove:
                _memory_saver.writes.pop(k, None)


@asynccontextmanager
async def get_checkpointer():
    """Provides a managed checkpointer using a thread-safe global connection pool or bounded MemorySaver."""
    global _pg_pool, _pg_setup_done

    if settings.database_url.startswith("postgres"):
        async with _pg_pool_lock:
            if _pg_pool is None:
                logger.info("Initializing global PostgreSQL connection pool...")
                _pg_pool = AsyncConnectionPool(
                    conninfo=settings.database_url,
                    max_size=20,
                    open=False,
                    kwargs={"autocommit": True, "prepare_threshold": 0},
                )
                await _pg_pool.open()
                _pg_setup_done = False

            checkpointer = AsyncPostgresSaver(_pg_pool)
            if not _pg_setup_done:
                await checkpointer.setup()
                _pg_setup_done = True

        yield checkpointer
    else:
        logger.debug("Using bounded singleton in-memory checkpointer.")
        yield _memory_saver


async def get_thread_checkpoint_state(thread_id: str) -> Optional[dict[str, Any]]:
    """Retrieves the latest checkpoint state for a given thread_id, redacting any API keys."""
    async with get_checkpointer() as checkpointer:
        graph = build_masc_graph(checkpointer=checkpointer)
        snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
        if not snapshot or not snapshot.values:
            return None

        safe_values: dict[str, Any] = {}
        for key, value in snapshot.values.items():
            if hasattr(value, "redacted_dump"):
                safe_values[key] = value.redacted_dump()
            elif hasattr(value, "model_dump"):
                safe_values[key] = value.model_dump()
            else:
                safe_values[key] = value

        return {
            "thread_id": thread_id,
            "next": list(snapshot.next) if snapshot.next else [],
            "values": safe_values,
        }


async def close_pg_pool():
    """Gracefully closes the global PostgreSQL connection pool."""
    global _pg_pool, _pg_setup_done
    async with _pg_pool_lock:
        if _pg_pool is not None:
            logger.info("Draining and closing global PostgreSQL connection pool...")
            await _pg_pool.close()
            _pg_pool = None
            _pg_setup_done = False
            logger.info("PostgreSQL connection pool successfully closed.")


def _normalize_Issue(issue: CritiqueItem | dict) -> CritiqueItem:
    """Ensures critique payload items are always CritiqueItem instances."""
    if isinstance(issue, CritiqueItem):
        return issue
    if isinstance(issue, dict):
        sev = str(issue.get("severity", "MEDIUM")).upper()
        if sev not in SEVERITY_RANK:
            sev = "MEDIUM"
        return CritiqueItem(
            severity=sev,  # type: ignore[arg-type]
            description=str(issue.get("description", "Unspecified issue")),
            recommendation=str(issue.get("recommendation", "Address the identified issue.")),
        )
    raise TypeError(f"Unsupported critique item type: {type(issue)}")


class MASCNodes:
    def __init__(self, sys_settings):
        self.settings = sys_settings

    def _get_llm(self, config: PersonaConfig):
        provider = config.llm_config.provider
        retry_config = {
            "stop_after_attempt": self.settings.llm_retry_attempts,
            "wait_exponential_jitter": False,
        }

        # Treat empty strings or placeholder "env" as None so SDKs fall back to environment variables
        raw_key = (config.llm_config.api_key or "").strip()
        resolved_api_key = None if raw_key in ("", "env") else raw_key

        # Validate custom base_url to prevent SSRF
        validated_base_url = validate_base_url(config.llm_config.base_url, self.settings)

        # Prevent server environment API key exfiltration to custom base_urls
        if validated_base_url and provider != "Ollama" and resolved_api_key is None:
            raise ValueError(
                f"Explicit API key is required when specifying a custom base_url for provider '{provider}'."
            )

        if provider == "Ollama":
            try:
                from langchain_ollama import ChatOllama
            except ImportError as exc:
                raise ImportError("Please run: pip install langchain-ollama") from exc

            return ChatOllama(
                model=config.llm_config.model_name,
                base_url=validated_base_url or "http://localhost:11434",
                temperature=config.llm_config.temperature,
                num_predict=config.llm_config.max_tokens,
            ).with_retry(**retry_config)

        provider_mapping = {
            "OpenAI": "openai",
            "Anthropic": "anthropic",
            "Google": "google_genai",
        }
        provider_str = provider_mapping.get(provider)
        if not provider_str:
            raise ValueError(f"Unsupported LLM Provider: {provider}")

        return init_chat_model(
            model=config.llm_config.model_name,
            model_provider=provider_str,
            temperature=config.llm_config.temperature,
            max_tokens=config.llm_config.max_tokens,
            api_key=resolved_api_key,
            base_url=validated_base_url,
        ).with_retry(**retry_config)

    async def propose(self, state: MASCState) -> dict:
        logger.info("Entering Proposer Node")
        config = state["config"].proposer
        llm = self._get_llm(config)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert {persona_name}. Your task is to generate a comprehensive, "
                    "well-structured, and well-supported initial response to the user's query. "
                    "Build the strongest possible initial case.",
                ),
                ("human", "{task_description}"),
            ]
        )

        chain = prompt | llm
        response = await chain.ainvoke(
            {
                "persona_name": config.persona_name,
                "task_description": state["task_description"],
            }
        )

        return {
            "current_artifact": Artifact(
                version="1.0_proposal",
                content=str(response.content),
                history=[f"Initial proposal by {config.persona_name} using {config.llm_config.model_name}"],
            ),
            "current_turn": 1,
        }

    async def adversarial_analysis(self, state: MASCState) -> dict:
        current_turn = state.get("current_turn", 1)
        logger.info(f"Entering Adversarial Analysis Node (Turn {current_turn})")
        current_content = state["current_artifact"].content
        task_description = state["task_description"]
        adversary_configs = state["config"].adversaries
        personas_registry = {**ADVERSARY_PERSONAS, **load_personas()}

        unknown_personas = [
            adv.persona_name for adv in adversary_configs if adv.persona_name not in personas_registry
        ]
        if unknown_personas:
            raise ValueError(
                f"Unknown adversary persona(s): {unknown_personas}. "
                f"Available personas: {list(personas_registry.keys())}"
            )

        critique_tasks = [
            self._run_single_adversary(
                adv_config=adv_config,
                name=adv_config.persona_name,
                persona=personas_registry[adv_config.persona_name],
                task_description=task_description,
                content=current_content,
            )
            for adv_config in adversary_configs
        ]

        results = await asyncio.gather(*critique_tasks, return_exceptions=True)

        valid_critiques: list[Critique] = []
        errors: list[str] = []
        for res in results:
            if isinstance(res, Critique):
                valid_critiques.append(res)
            elif isinstance(res, Exception):
                logger.error(f"Adversary task failed with exception: {res}\n{traceback.format_exc()}")
                errors.append(str(res))

        if adversary_configs and not valid_critiques:
            raise RuntimeError(
                f"All configured adversaries failed during Turn {current_turn}. Errors: {'; '.join(errors)}"
            )

        return {
            "critiques_collection": CritiquesCollection(critiques=valid_critiques),
            "current_turn": current_turn,
        }

    async def _run_single_adversary(
        self,
        adv_config: PersonaConfig,
        name: str,
        persona: dict,
        task_description: str,
        content: str,
    ) -> Critique:
        try:
            llm = self._get_llm(adv_config)
            structured_llm = llm.with_structured_output(CritiqueLLMOutput)

            # Pass system_prompt as an input variable so literal curly braces in custom prompts never raise KeyError
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "{system_prompt}"),
                    (
                        "human",
                        "Original Task Description:\n"
                        "<task_description>\n{task_description}\n</task_description>\n\n"
                        "Artifact to Audit:\n"
                        "<artifact>\n{artifact_content}\n</artifact>",
                    ),
                ]
            )

            chain = prompt | structured_llm
            logger.info(f"Running adversary: {name}")

            result: CritiqueLLMOutput = await chain.ainvoke(
                {
                    "system_prompt": persona["system_prompt"],
                    "task_description": task_description,
                    "artifact_content": content,
                }
            )
            logger.info(f"Successfully generated critique from {name}")

            payload = result.payload if result and result.payload else []
            return Critique(
                source_adversary=name,
                critique_type=persona["critique_type"],
                payload=payload,
            )
        except Exception as e:
            logger.error(f"Adversary '{name}' failed. Error: {e}")
            raise

    async def triage_critiques(self, state: MASCState) -> dict:
        current_turn = state.get("current_turn", 1)
        logger.info(f"Entering Triage Node (Turn {current_turn})")

        collection = state.get("critiques_collection")
        if not collection or not collection.critiques:
            return {
                "critiques_collection": CritiquesCollection(critiques=[]),
                "current_turn": current_turn,
            }

        sorted_critiques: list[Critique] = []
        for critique in collection.critiques:
            if not critique.payload:
                continue
            normalized_payload = [_normalize_Issue(item) for item in critique.payload]
            sorted_payload = sorted(
                normalized_payload,
                key=lambda item: SEVERITY_RANK.get(item.severity, 99),
            )
            sorted_critiques.append(
                Critique(
                    source_adversary=critique.source_adversary,
                    critique_type=critique.critique_type,
                    payload=sorted_payload,
                )
            )

        def _critique_sort_key(c: Critique) -> tuple[int, int]:
            type_rank = 0 if c.critique_type == "ANTAGONISTIC" else 1
            highest_severity = min(
                (SEVERITY_RANK.get(item.severity, 99) for item in c.payload),
                default=99,
            )
            return type_rank, highest_severity

        sorted_critiques.sort(key=_critique_sort_key)
        logger.info("Deterministic triage complete. Processing order established.")

        return {
            "critiques_collection": CritiquesCollection(critiques=sorted_critiques),
            "current_turn": current_turn,
        }

    async def synthesize_sequential(self, state: MASCState) -> dict:
        current_turn = state.get("current_turn", 1)
        logger.info(f"Entering Synthesizer Node (Sequential Refinement - Turn {current_turn})")
        config = state["config"].synthesizer
        llm = self._get_llm(config)

        current_artifact = state["current_artifact"].model_copy(deep=True)
        history_log = current_artifact.history.copy()
        task_description = state["task_description"]
        critiques_collection = state.get("critiques_collection") or CritiquesCollection(critiques=[])

        actionable_critiques = [c for c in critiques_collection.critiques if c.payload]
        if not actionable_critiques:
            history_log.append(f"[Turn {current_turn}] No actionable critiques required changes.")
            return {
                "current_artifact": Artifact(
                    version=f"{current_turn + 1}.0_sequential",
                    content=current_artifact.content,
                    history=history_log,
                ),
                "current_turn": current_turn + 1,
            }

        successful_steps = 0
        errors: list[str] = []
        turn_modifications: list[str] = []

        for critique in actionable_critiques:
            prior_context = (
                "\n".join(f"- {m}" for m in turn_modifications)
                if turn_modifications
                else "None yet in this turn."
            )
            try:
                if critique.critique_type == "ANTAGONISTIC":
                    logger.info(f"Refuting antagonistic critique from {critique.source_adversary}")
                    refutation_prompt = ChatPromptTemplate.from_messages(
                        [
                            (
                                "system",
                                "You are a Master Synthesizer. Your task is to harden a proposal against this cynical, "
                                "antagonistic argument while remaining strictly faithful to the Original Task Description. "
                                "Identify the core fallacious principle in the critique and revise the proposal to "
                                "explicitly defend against it. Add justifications or foundational principles that make "
                                "the artifact's core thesis more robust. Do not integrate the destructive premise of "
                                "the antagonistic feedback; build a defense against it. Preserve any fixes already "
                                "applied earlier in this turn.",
                            ),
                            (
                                "human",
                                "Original Task Description:\n{task_description}\n\n"
                                "Fixes Already Applied Earlier in This Turn:\n{prior_context}\n\n"
                                "Current Proposal:\n{proposal}\n\n"
                                "Antagonistic Critique:\n{ant_critique}",
                            ),
                        ]
                    )
                    refutation_chain = refutation_prompt | llm
                    response = await refutation_chain.ainvoke(
                        {
                            "task_description": task_description,
                            "prior_context": prior_context,
                            "proposal": current_artifact.content,
                            "ant_critique": critique.model_dump_json(indent=2),
                        }
                    )
                    current_artifact.content = str(response.content)
                    entry = f"[Turn {current_turn}] Hardened against: {critique.source_adversary}"
                    history_log.append(entry)
                    turn_modifications.append(entry)
                    successful_steps += 1

                elif critique.critique_type == "CONSTRUCTIVE":
                    normalized_issues = [_normalize_Issue(item) for item in critique.payload]
                    logger.info(
                        f"Addressing {len(normalized_issues)} constructive issue(s) from {critique.source_adversary}"
                    )
                    formatted_issues = "\n".join(
                        f"{idx}. [{issue.severity}] Issue: {issue.description}\n"
                        f"   Recommendation: {issue.recommendation}"
                        for idx, issue in enumerate(normalized_issues, start=1)
                    )

                    revision_prompt = ChatPromptTemplate.from_messages(
                        [
                            (
                                "system",
                                "You are a Master Technical Editor. Your task is to revise the current draft to "
                                "address the prioritized constructive critiques from the specialist auditor while "
                                "remaining strictly aligned with the Original Task Description. Ensure all listed "
                                "issues are resolved cleanly without degrading unaffected sections of the draft or "
                                "reverting fixes already applied earlier in this turn.",
                            ),
                            (
                                "human",
                                "Original Task Description:\n{task_description}\n\n"
                                "Fixes Already Applied Earlier in This Turn:\n{prior_context}\n\n"
                                "Current Draft:\n{draft}\n\n"
                                "Auditor ({adversary_name}) Constructive Critiques:\n{formatted_issues}",
                            ),
                        ]
                    )
                    revision_chain = revision_prompt | llm
                    response = await revision_chain.ainvoke(
                        {
                            "task_description": task_description,
                            "prior_context": prior_context,
                            "draft": current_artifact.content,
                            "adversary_name": critique.source_adversary,
                            "formatted_issues": formatted_issues,
                        }
                    )
                    current_artifact.content = str(response.content)
                    for issue in normalized_issues:
                        entry = (
                            f"[Turn {current_turn}] Fixed [{issue.severity}] critique: "
                            f"'{issue.description}' from {critique.source_adversary}."
                        )
                        history_log.append(entry)
                        turn_modifications.append(entry)
                    successful_steps += 1

            except Exception as e:
                logger.error(f"Failed to process critique from {critique.source_adversary}. Error: {e}")
                errors.append(f"{critique.source_adversary}: {e}")
                history_log.append(
                    f"[Turn {current_turn}] ERROR: Failed to apply fix for critique from {critique.source_adversary}."
                )

        if actionable_critiques and successful_steps == 0:
            raise RuntimeError(
                f"All sequential synthesis steps failed during Turn {current_turn}. Errors: {'; '.join(errors)}"
            )

        final_artifact = Artifact(
            version=f"{current_turn + 1}.0_sequential",
            content=current_artifact.content,
            history=history_log,
        )
        return {"current_artifact": final_artifact, "current_turn": current_turn + 1}

    async def synthesize_architect(self, state: MASCState) -> dict:
        current_turn = state.get("current_turn", 1)
        logger.info(f"Entering Synthesizer Node (Architect Protocol - Turn {current_turn})")
        config = state["config"].synthesizer
        llm = self._get_llm(config)

        current_artifact = state["current_artifact"]
        critiques_collection = state.get("critiques_collection") or CritiquesCollection(critiques=[])
        task_description = state["task_description"]

        history_log = current_artifact.history.copy()
        history_log.append(
            f"[Turn {current_turn}] Architect holistic review of {len(critiques_collection.critiques)} critiques."
        )

        architect_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a Chief AI Architect. You are presented with the original task description, the current "
                    "proposal, and a collection of adversarial critiques. Your task is to analyze all critiques "
                    "simultaneously, resolve conflicting advice, refute antagonistic challenges, and rewrite the "
                    "proposal from the ground up to create a single, unified, and hardened document that fulfills "
                    "the original task description.",
                ),
                (
                    "human",
                    "Original Task Description:\n{task_description}\n\n"
                    "Current Proposal:\n{proposal}\n\n"
                    "Adversarial Critiques:\n{critiques}",
                ),
            ]
        )

        architect_chain = architect_prompt | llm
        response = await architect_chain.ainvoke(
            {
                "task_description": task_description,
                "proposal": current_artifact.content,
                "critiques": critiques_collection.model_dump_json(indent=2),
            }
        )

        history_log.append(f"[Turn {current_turn}] Architect unified critiques into a hardened artifact.")
        final_artifact = Artifact(
            version=f"{current_turn + 1}.0_architect",
            content=str(response.content),
            history=history_log,
        )

        return {"current_artifact": final_artifact, "current_turn": current_turn + 1}


def should_route_to_synthesis(state: MASCState) -> Literal["triage_critiques", "synthesize_architect"]:
    protocol = state["config"].synthesis_protocol
    logger.info(f"Routing to synthesis protocol: {protocol}")
    if protocol == "Architect":
        return "synthesize_architect"
    return "triage_critiques"


def should_continue_loop(state: MASCState) -> Literal["adversarial_analysis", "__end__"]:
    if state["current_turn"] > state["config"].max_turns:
        logger.info("Max turns reached. Ending workflow.")
        return "__end__"

    # Early convergence: if critiques_collection is populated for this turn, check if any actionable issues existed
    collection = state.get("critiques_collection")
    if collection is not None:
        has_actionable_issues = any(len(c.payload) > 0 for c in collection.critiques)
        if not has_actionable_issues:
            logger.info("Early convergence achieved: no actionable critiques remaining. Ending workflow.")
            return "__end__"

    logger.info(f"Looping back for Turn {state['current_turn']} / {state['config'].max_turns}")
    return "adversarial_analysis"


def build_masc_graph(checkpointer=None) -> CompiledStateGraph:
    """Builds the MASC graph, dynamically binding it to the provided checkpointer."""
    nodes = MASCNodes(settings)
    graph_builder = StateGraph(MASCState)

    graph_builder.add_node("propose", nodes.propose)
    graph_builder.add_node("adversarial_analysis", nodes.adversarial_analysis)
    graph_builder.add_node("triage_critiques", nodes.triage_critiques)
    graph_builder.add_node("synthesize_sequential", nodes.synthesize_sequential)
    graph_builder.add_node("synthesize_architect", nodes.synthesize_architect)

    graph_builder.set_entry_point("propose")
    graph_builder.add_edge("propose", "adversarial_analysis")

    graph_builder.add_conditional_edges(
        "adversarial_analysis",
        should_route_to_synthesis,
        {
            "triage_critiques": "triage_critiques",
            "synthesize_architect": "synthesize_architect",
        },
    )

    graph_builder.add_edge("triage_critiques", "synthesize_sequential")

    graph_builder.add_conditional_edges(
        "synthesize_sequential",
        should_continue_loop,
        {"adversarial_analysis": "adversarial_analysis", "__end__": END},
    )
    graph_builder.add_conditional_edges(
        "synthesize_architect",
        should_continue_loop,
        {"adversarial_analysis": "adversarial_analysis", "__end__": END},
    )

    return graph_builder.compile(checkpointer=checkpointer)


async def execute_masc_workflow(task: str, config: MASCConfig, thread_id: str) -> AsyncGenerator[dict, None]:
    """Executes the MASC workflow and streams node state updates."""
    initial_state = {"task_description": task, "config": config, "current_turn": 1}
    run_config: dict = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": settings.max_recursion_depth,
    }

    langfuse_handler = None
    if settings.langfuse_public_key and settings.langfuse_secret_key:
        try:
            from langfuse.callback import CallbackHandler

            langfuse_handler = CallbackHandler(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
            run_config["callbacks"] = [langfuse_handler]
            logger.info("Langfuse observability enabled for this execution.")
        except ImportError:
            logger.warning("Langfuse keys found, but langfuse package is not installed. Run 'pip install langfuse'.")

    if not settings.database_url.startswith("postgres"):
        _track_and_prune_memory_threads(thread_id)

    try:
        async with get_checkpointer() as checkpointer:
            graph = build_masc_graph(checkpointer=checkpointer)
            async for state in graph.astream(initial_state, run_config):
                yield state
    finally:
        if langfuse_handler is not None and hasattr(langfuse_handler, "flush"):
            try:
                langfuse_handler.flush()
            except Exception as e:
                logger.debug(f"Failed to flush Langfuse handler: {e}")
