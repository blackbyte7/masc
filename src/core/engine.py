import asyncio
import json
import logging
import traceback
from collections import deque
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Literal, Optional

from langchain.chat_models import init_chat_model
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from psycopg_pool import AsyncConnectionPool

from src.config.settings import settings, validate_base_url
from src.core.personas import (
    ADVERSARY_PERSONAS,
    load_personas,
    resolve_adversary_persona,
    run_deterministic_verification,
)
from src.core.state import (
    Artifact,
    Critique,
    CritiqueItem,
    CritiqueLLMOutput,
    CritiquesCollection,
    MASCConfig,
    MASCState,
    PersonaConfig,
    PersonaRouterOutput,
    TokenUsage,
)

logger = logging.getLogger(__name__)

_pg_pool: Optional[AsyncConnectionPool] = None
_pg_pool_lock = asyncio.Lock()
_pg_setup_done: bool = False
_memory_saver = MemorySaver()
_memory_thread_order: deque[str] = deque()
_llm_semaphore: Optional[asyncio.Semaphore] = None
_llm_semaphore_limit: Optional[int] = None

SEVERITY_RANK = {
    "CRITICAL": 0,
    "HIGH": 1,
    "MEDIUM": 2,
    "LOW": 3,
}


def _get_llm_semaphore(max_concurrent: int) -> asyncio.Semaphore:
    """Returns a process-wide asyncio.Semaphore for bounding simultaneous LLM calls."""
    global _llm_semaphore, _llm_semaphore_limit
    if _llm_semaphore is None or _llm_semaphore_limit != max_concurrent:
        _llm_semaphore = asyncio.Semaphore(max(1, max_concurrent))
        _llm_semaphore_limit = max_concurrent
    return _llm_semaphore


class _TokenUsageCallback(AsyncCallbackHandler):
    """Captures token usage metadata across standard and structured LLM calls."""

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        if response.llm_output and isinstance(response.llm_output, dict):
            token_usage = response.llm_output.get("token_usage") or response.llm_output.get("usage") or {}
            if isinstance(token_usage, dict):
                self.input_tokens += int(token_usage.get("prompt_tokens", token_usage.get("input_tokens", 0)) or 0)
                self.output_tokens += int(
                    token_usage.get("completion_tokens", token_usage.get("output_tokens", 0)) or 0
                )
                self.total_tokens += int(token_usage.get("total_tokens", 0) or 0)

        for gen_list in response.generations:
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                usage_meta = getattr(msg, "usage_metadata", None)
                if isinstance(usage_meta, dict):
                    self.input_tokens += int(usage_meta.get("input_tokens", 0) or 0)
                    self.output_tokens += int(usage_meta.get("output_tokens", 0) or 0)
                    self.total_tokens += int(usage_meta.get("total_tokens", 0) or 0)


def _extract_usage(response: Any, cb: _TokenUsageCallback) -> tuple[int, int, int]:
    """Extracts (input_tokens, output_tokens, total_tokens) from either callback or AIMessage."""
    if cb.total_tokens > 0 or cb.input_tokens > 0 or cb.output_tokens > 0:
        total = cb.total_tokens or (cb.input_tokens + cb.output_tokens)
        return cb.input_tokens, cb.output_tokens, total

    usage_meta = getattr(response, "usage_metadata", None)
    if isinstance(usage_meta, dict):
        inp = int(usage_meta.get("input_tokens", 0) or 0)
        out = int(usage_meta.get("output_tokens", 0) or 0)
        tot = int(usage_meta.get("total_tokens", 0) or (inp + out))
        return inp, out, tot

    return 0, 0, 0


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

    async def _invoke_chain(self, chain: Any, inputs: dict[str, Any]) -> tuple[Any, tuple[int, int, int]]:
        """Executes a LangChain runnable under the global concurrency semaphore and records token usage."""
        sem = _get_llm_semaphore(self.settings.max_concurrent_llm_calls)
        usage_cb = _TokenUsageCallback()
        async with sem:
            response = await chain.ainvoke(inputs, config={"callbacks": [usage_cb]})
        return response, _extract_usage(response, usage_cb)

    async def _apply_response_schema_if_needed(
        self,
        llm: Any,
        task_description: str,
        draft_content: str,
        response_schema: Optional[dict[str, Any]],
        current_usage: TokenUsage,
    ) -> tuple[str, TokenUsage]:
        """Formats the synthesized artifact to strictly conform to config.response_schema when requested."""
        if not response_schema:
            return draft_content, current_usage

        logger.info("Enforcing structured JSON response_schema on synthesized artifact.")
        structured_llm = llm.with_structured_output(response_schema)
        format_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Convert the following hardened proposal into the required structured JSON schema while "
                    "preserving all technical details, security mitigations, and justifications.",
                ),
                (
                    "human",
                    "Original Task Description:\n{task_description}\n\nHardened Artifact:\n{draft_content}",
                ),
            ]
        )
        chain = format_prompt | structured_llm
        structured_res, (inp, out, tot) = await self._invoke_chain(
            chain,
            {"task_description": task_description, "draft_content": draft_content},
        )
        updated_usage = current_usage.add(inp, out, tot, calls=1)
        if isinstance(structured_res, (dict, list)):
            return json.dumps(structured_res, indent=2), updated_usage
        if hasattr(structured_res, "model_dump_json"):
            return structured_res.model_dump_json(indent=2), updated_usage
        return str(structured_res), updated_usage

    async def propose(self, state: MASCState) -> dict:
        logger.info("Entering Proposer Node")
        config = state["config"].proposer
        llm = self._get_llm(config)
        current_usage = state.get("token_usage") or TokenUsage()

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
        response, (inp, out, tot) = await self._invoke_chain(
            chain,
            {
                "persona_name": config.persona_name,
                "task_description": state["task_description"],
            },
        )

        return {
            "current_artifact": Artifact(
                version="1.0_proposal",
                content=str(response.content),
                history=[f"Initial proposal by {config.persona_name} using {config.llm_config.model_name}"],
            ),
            "current_turn": 1,
            "token_usage": current_usage.add(inp, out, tot, calls=1),
        }

    async def route_personas(self, state: MASCState) -> dict:
        """
        Optional Auto-Persona Router node: inspects the task description and available persona registry
        to dynamically select the best adversary personas when config.auto_select_adversaries is True.
        """
        current_turn = state.get("current_turn", 1)
        masc_cfg = state["config"].model_copy(deep=True)
        current_usage = state.get("token_usage") or TokenUsage()

        if not masc_cfg.auto_select_adversaries:
            return {"config": masc_cfg, "current_turn": current_turn, "token_usage": current_usage}

        logger.info("Entering Optional Auto-Persona Router Node")
        personas_registry = {**ADVERSARY_PERSONAS, **load_personas()}
        available_descriptions = "\n".join(
            f"- {name} ({meta['critique_type']}): {meta['system_prompt'][:220]}..."
            for name, meta in personas_registry.items()
        )

        selected_names: list[str] = []
        try:
            llm = self._get_llm(masc_cfg.synthesizer)
            structured_router = llm.with_structured_output(PersonaRouterOutput)
            router_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are an Chief Audit Router. Select between 1 and 3 adversary personas from the "
                        "Available Personas list that are best suited to critique and harden the user's task.",
                    ),
                    (
                        "human",
                        "Available Personas:\n{available_descriptions}\n\n"
                        "Task Description:\n{task_description}",
                    ),
                ]
            )
            chain = router_prompt | structured_router
            router_res, (inp, out, tot) = await self._invoke_chain(
                chain,
                {
                    "available_descriptions": available_descriptions,
                    "task_description": state["task_description"],
                },
            )
            current_usage = current_usage.add(inp, out, tot, calls=1)
            if router_res and router_res.selected_personas:
                selected_names = [p for p in router_res.selected_personas if p in personas_registry]
        except Exception as e:
            logger.warning(f"Auto-Persona Router LLM call failed ({e}); using deterministic fallback.")

        if not selected_names:
            # Deterministic fallback if LLM router fails or returns empty
            fallback_candidates = ["DA", "UncertaintyQuantifier", "CodeAuditor"]
            selected_names = [p for p in fallback_candidates if p in personas_registry][:2]

        existing_names = {adv.persona_name for adv in masc_cfg.adversaries}
        default_adv_llm = masc_cfg.synthesizer.llm_config.model_copy(deep=True)
        for name in selected_names:
            if len(masc_cfg.adversaries) >= self.settings.max_adversaries:
                break
            if name not in existing_names:
                masc_cfg.adversaries.append(PersonaConfig(persona_name=name, llm_config=default_adv_llm))
                existing_names.add(name)

        logger.info(f"Auto-Persona Router configured adversaries: {[a.persona_name for a in masc_cfg.adversaries]}")
        return {
            "config": masc_cfg,
            "current_turn": current_turn,
            "token_usage": current_usage,
        }

    async def adversarial_analysis(self, state: MASCState) -> dict:
        current_turn = state.get("current_turn", 1)
        logger.info(f"Entering Adversarial Analysis Node (Turn {current_turn})")
        current_content = state["current_artifact"].content
        task_description = state["task_description"]
        masc_cfg = state["config"]
        adversary_configs = masc_cfg.adversaries
        personas_registry = {**ADVERSARY_PERSONAS, **load_personas()}
        current_usage = state.get("token_usage") or TokenUsage()

        resolved_personas = [
            resolve_adversary_persona(adv_config, personas_registry) for adv_config in adversary_configs
        ]

        critique_tasks = [
            self._run_single_adversary(
                adv_config=adv_config,
                name=adv_config.persona_name,
                persona=persona_def,
                task_description=task_description,
                content=current_content,
                enable_verification=masc_cfg.enable_verification_tools,
            )
            for adv_config, persona_def in zip(adversary_configs, resolved_personas)
        ]

        results = await asyncio.gather(*critique_tasks, return_exceptions=True)

        valid_critiques: list[Critique] = []
        errors: list[str] = []
        for res in results:
            if isinstance(res, tuple) and len(res) == 2 and isinstance(res[0], Critique):
                critique_obj, (inp, out, tot) = res
                valid_critiques.append(critique_obj)
                current_usage = current_usage.add(inp, out, tot, calls=1)
            elif isinstance(res, Critique):
                valid_critiques.append(res)
                current_usage = current_usage.add(calls=1)
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
            "token_usage": current_usage,
        }

    async def _run_single_adversary(
        self,
        adv_config: PersonaConfig,
        name: str,
        persona: dict,
        task_description: str,
        content: str,
        enable_verification: bool = False,
    ) -> tuple[Critique, tuple[int, int, int]]:
        try:
            llm = self._get_llm(adv_config)
            structured_llm = llm.with_structured_output(CritiqueLLMOutput)

            verification_section = ""
            if enable_verification:
                findings = run_deterministic_verification(name, content)
                if findings:
                    formatted_findings = "\n".join(f"- {f}" for f in findings)
                    verification_section = (
                        "\n\nDeterministic Static Verification Findings (must be addressed if applicable):\n"
                        f"<deterministic_findings>\n{formatted_findings}\n</deterministic_findings>"
                    )

            # Pass system_prompt as an input variable so literal curly braces in custom prompts never raise KeyError
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "{system_prompt}"),
                    (
                        "human",
                        "Original Task Description:\n"
                        "<task_description>\n{task_description}\n</task_description>\n\n"
                        "Artifact to Audit:\n"
                        "<artifact>\n{artifact_content}\n</artifact>"
                        "{verification_section}",
                    ),
                ]
            )

            chain = prompt | structured_llm
            logger.info(f"Running adversary: {name}")

            result, usage_tuple = await self._invoke_chain(
                chain,
                {
                    "system_prompt": persona["system_prompt"],
                    "task_description": task_description,
                    "artifact_content": content,
                    "verification_section": verification_section,
                },
            )
            logger.info(f"Successfully generated critique from {name}")

            payload = result.payload if result and result.payload else []
            critique = Critique(
                source_adversary=name,
                critique_type=persona["critique_type"],
                payload=payload,
            )
            return critique, usage_tuple
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
            return (type_rank, highest_severity)

        sorted_critiques.sort(key=_critique_sort_key)
        logger.info("Deterministic triage complete. Processing order established.")

        return {
            "critiques_collection": CritiquesCollection(critiques=sorted_critiques),
            "current_turn": current_turn,
        }

    async def synthesize_sequential(self, state: MASCState) -> dict:
        current_turn = state.get("current_turn", 1)
        logger.info(f"Entering Synthesizer Node (Sequential Refinement - Turn {current_turn})")
        masc_cfg = state["config"]
        config = masc_cfg.synthesizer
        llm = self._get_llm(config)
        current_usage = state.get("token_usage") or TokenUsage()

        current_artifact = state["current_artifact"].model_copy(deep=True)
        history_log = current_artifact.history.copy()
        task_description = state["task_description"]
        critiques_collection = state.get("critiques_collection") or CritiquesCollection(critiques=[])

        actionable_critiques = [c for c in critiques_collection.critiques if c.payload]
        if not actionable_critiques:
            history_log.append(f"[Turn {current_turn}] No actionable critiques required changes.")
            formatted_content, current_usage = await self._apply_response_schema_if_needed(
                llm, task_description, current_artifact.content, masc_cfg.response_schema, current_usage
            )
            return {
                "current_artifact": Artifact(
                    version=f"{current_turn + 1}.0_sequential",
                    content=formatted_content,
                    history=history_log,
                ),
                "current_turn": current_turn + 1,
                "token_usage": current_usage,
            }

        successful_steps = 0
        errors: list[str] = []
        turn_modifications: list[str] = []

        antagonistic_critiques = [c for c in actionable_critiques if c.critique_type == "ANTAGONISTIC"]
        constructive_critiques = [c for c in actionable_critiques if c.critique_type == "CONSTRUCTIVE"]

        # Stage 1: Process Antagonistic Critiques
        for critique in antagonistic_critiques:
            prior_context = (
                "\n".join(f"- {m}" for m in turn_modifications)
                if turn_modifications
                else "None yet in this turn."
            )
            try:
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
                response, (inp, out, tot) = await self._invoke_chain(
                    refutation_chain,
                    {
                        "task_description": task_description,
                        "prior_context": prior_context,
                        "proposal": current_artifact.content,
                        "ant_critique": critique.model_dump_json(indent=2),
                    },
                )
                current_usage = current_usage.add(inp, out, tot, calls=1)
                current_artifact.content = str(response.content)
                entry = f"[Turn {current_turn}] Hardened against: {critique.source_adversary}"
                history_log.append(entry)
                turn_modifications.append(entry)
                successful_steps += 1
            except Exception as e:
                logger.error(f"Failed to process critique from {critique.source_adversary}. Error: {e}")
                errors.append(f"{critique.source_adversary}: {e}")
                history_log.append(
                    f"[Turn {current_turn}] ERROR: Failed to apply fix for critique from {critique.source_adversary}."
                )

        # Stage 2: Process Constructive Critiques (either batched in a single pass or per-adversary)
        if constructive_critiques and masc_cfg.batch_constructive_critiques:
            prior_context = (
                "\n".join(f"- {m}" for m in turn_modifications)
                if turn_modifications
                else "None yet in this turn."
            )
            all_constructive_lines: list[str] = []
            log_entries_to_add: list[str] = []
            idx_counter = 1
            for critique in constructive_critiques:
                normalized_issues = [_normalize_Issue(item) for item in critique.payload]
                for issue in normalized_issues:
                    all_constructive_lines.append(
                        f"{idx_counter}. [{issue.severity}] (Source: {critique.source_adversary}) "
                        f"Issue: {issue.description}\n   Recommendation: {issue.recommendation}"
                    )
                    log_entries_to_add.append(
                        f"[Turn {current_turn}] Fixed [{issue.severity}] critique: "
                        f"'{issue.description}' from {critique.source_adversary}."
                    )
                    idx_counter += 1

            try:
                revision_prompt = ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            "You are a Master Technical Editor. Your task is to revise the current draft to "
                            "address all prioritized constructive critiques from the specialist auditors in a single "
                            "unified pass while remaining strictly aligned with the Original Task Description. "
                            "Ensure all listed issues are resolved cleanly without degrading unaffected sections "
                            "or reverting fixes already applied earlier in this turn.",
                        ),
                        (
                            "human",
                            "Original Task Description:\n{task_description}\n\n"
                            "Fixes Already Applied Earlier in This Turn:\n{prior_context}\n\n"
                            "Current Draft:\n{draft}\n\n"
                            "Prioritized Constructive Critiques:\n{formatted_issues}",
                        ),
                    ]
                )
                revision_chain = revision_prompt | llm
                response, (inp, out, tot) = await self._invoke_chain(
                    revision_chain,
                    {
                        "task_description": task_description,
                        "prior_context": prior_context,
                        "draft": current_artifact.content,
                        "formatted_issues": "\n".join(all_constructive_lines),
                    },
                )
                current_usage = current_usage.add(inp, out, tot, calls=1)
                current_artifact.content = str(response.content)
                history_log.extend(log_entries_to_add)
                turn_modifications.extend(log_entries_to_add)
                successful_steps += 1
            except Exception as e:
                logger.error(f"Failed batched constructive synthesis. Error: {e}")
                errors.append(f"BatchedConstructive: {e}")
                history_log.append(f"[Turn {current_turn}] ERROR: Failed to apply batched constructive fixes.")
        else:
            for critique in constructive_critiques:
                prior_context = (
                    "\n".join(f"- {m}" for m in turn_modifications)
                    if turn_modifications
                    else "None yet in this turn."
                )
                try:
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
                    response, (inp, out, tot) = await self._invoke_chain(
                        revision_chain,
                        {
                            "task_description": task_description,
                            "prior_context": prior_context,
                            "draft": current_artifact.content,
                            "adversary_name": critique.source_adversary,
                            "formatted_issues": formatted_issues,
                        },
                    )
                    current_usage = current_usage.add(inp, out, tot, calls=1)
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

        if masc_cfg.response_schema:
            current_artifact.content, current_usage = await self._apply_response_schema_if_needed(
                llm, task_description, current_artifact.content, masc_cfg.response_schema, current_usage
            )
            history_log.append(f"[Turn {current_turn}] Enforced structured JSON response_schema.")

        final_artifact = Artifact(
            version=f"{current_turn + 1}.0_sequential",
            content=current_artifact.content,
            history=history_log,
        )
        return {
            "current_artifact": final_artifact,
            "current_turn": current_turn + 1,
            "token_usage": current_usage,
        }

    async def synthesize_architect(self, state: MASCState) -> dict:
        current_turn = state.get("current_turn", 1)
        logger.info(f"Entering Synthesizer Node (Architect Protocol - Turn {current_turn})")
        masc_cfg = state["config"]
        config = masc_cfg.synthesizer
        llm = self._get_llm(config)
        current_usage = state.get("token_usage") or TokenUsage()

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
        response, (inp, out, tot) = await self._invoke_chain(
            architect_chain,
            {
                "task_description": task_description,
                "proposal": current_artifact.content,
                "critiques": critiques_collection.model_dump_json(indent=2),
            },
        )
        current_usage = current_usage.add(inp, out, tot, calls=1)
        synthesized_text = str(response.content)

        if masc_cfg.response_schema:
            synthesized_text, current_usage = await self._apply_response_schema_if_needed(
                llm, task_description, synthesized_text, masc_cfg.response_schema, current_usage
            )
            history_log.append(f"[Turn {current_turn}] Enforced structured JSON response_schema.")

        history_log.append(f"[Turn {current_turn}] Architect unified critiques into a hardened artifact.")
        final_artifact = Artifact(
            version=f"{current_turn + 1}.0_architect",
            content=synthesized_text,
            history=history_log,
        )

        return {
            "current_artifact": final_artifact,
            "current_turn": current_turn + 1,
            "token_usage": current_usage,
        }


def should_propose_or_audit(
    state: MASCState,
) -> Literal["propose", "route_personas", "adversarial_analysis"]:
    """Routes directly to audit (or optional persona router) when an initial artifact is supplied."""
    artifact = state.get("current_artifact")
    if artifact and artifact.content and artifact.content.strip():
        logger.info("Initial artifact detected; skipping Proposer.")
        if state["config"].auto_select_adversaries:
            return "route_personas"
        return "adversarial_analysis"
    return "propose"


def should_route_after_propose(
    state: MASCState,
) -> Literal["route_personas", "adversarial_analysis"]:
    """Routes to the optional Auto-Persona Router only when config.auto_select_adversaries is True."""
    if state["config"].auto_select_adversaries:
        return "route_personas"
    return "adversarial_analysis"


def should_route_to_synthesis(state: MASCState) -> Literal["triage_critiques", "synthesize_architect"]:
    protocol = state["config"].synthesis_protocol
    logger.info(f"Routing to synthesis protocol: {protocol}")
    if protocol == "Architect":
        return "synthesize_architect"
    return "triage_critiques"


def should_continue_loop(state: MASCState) -> Literal["adversarial_analysis", "__end__"]:
    config = state["config"]
    if state["current_turn"] > config.max_turns:
        logger.info("Max turns reached. Ending workflow.")
        return "__end__"

    # FinOps circuit breaker: check if cumulative token budget has been reached
    token_usage = state.get("token_usage")
    if config.max_total_tokens is not None and token_usage is not None:
        if token_usage.total_tokens >= config.max_total_tokens:
            logger.warning(
                f"Token budget reached ({token_usage.total_tokens} >= {config.max_total_tokens}). "
                "Halting workflow early."
            )
            return "__end__"

    # Early convergence: check if any critiques meet or exceed the configured stop_on_max_severity threshold
    collection = state.get("critiques_collection")
    if collection is not None and config.stop_on_max_severity != "NONE":
        threshold_rank = SEVERITY_RANK.get(config.stop_on_max_severity, 3)
        has_qualifying_issues = any(
            SEVERITY_RANK.get(item.severity, 99) <= threshold_rank
            for critique in collection.critiques
            for item in critique.payload
        )
        if not has_qualifying_issues:
            logger.info(
                f"Early convergence achieved: no critiques with severity >= {config.stop_on_max_severity}. "
                "Ending workflow."
            )
            return "__end__"

    logger.info(f"Looping back for Turn {state['current_turn']} / {config.max_turns}")
    return "adversarial_analysis"


def build_masc_graph(checkpointer=None, interrupt_for_approval: bool = False) -> CompiledStateGraph:
    """Builds the MASC graph, dynamically binding it to the provided checkpointer and optional HITL breakpoints."""
    nodes = MASCNodes(settings)
    graph_builder = StateGraph(MASCState)

    graph_builder.add_node("propose", nodes.propose)
    graph_builder.add_node("route_personas", nodes.route_personas)
    graph_builder.add_node("adversarial_analysis", nodes.adversarial_analysis)
    graph_builder.add_node("triage_critiques", nodes.triage_critiques)
    graph_builder.add_node("synthesize_sequential", nodes.synthesize_sequential)
    graph_builder.add_node("synthesize_architect", nodes.synthesize_architect)

    graph_builder.set_conditional_entry_point(
        should_propose_or_audit,
        {
            "propose": "propose",
            "route_personas": "route_personas",
            "adversarial_analysis": "adversarial_analysis",
        },
    )
    graph_builder.add_conditional_edges(
        "propose",
        should_route_after_propose,
        {
            "route_personas": "route_personas",
            "adversarial_analysis": "adversarial_analysis",
        },
    )
    graph_builder.add_edge("route_personas", "adversarial_analysis")

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

    interrupt_nodes = (
        ["synthesize_sequential", "synthesize_architect"]
        if (interrupt_for_approval and checkpointer is not None)
        else None
    )
    return graph_builder.compile(checkpointer=checkpointer, interrupt_before=interrupt_nodes)


async def execute_masc_workflow(
    task: str,
    config: MASCConfig,
    thread_id: str,
    initial_artifact: Optional[str] = None,
) -> AsyncGenerator[dict, None]:
    """Executes the MASC workflow and streams node state updates."""
    resolved_initial_text = (initial_artifact or config.initial_artifact or "").strip()
    initial_state: dict[str, Any] = {
        "task_description": task,
        "config": config,
        "current_turn": 1,
        "token_usage": TokenUsage(),
    }
    if resolved_initial_text:
        initial_state["current_artifact"] = Artifact(
            version="1.0_initial",
            content=resolved_initial_text,
            history=["User-supplied initial artifact (Audit-Only mode)"],
        )

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
            graph = build_masc_graph(
                checkpointer=checkpointer,
                interrupt_for_approval=config.require_human_approval,
            )
            async for state in graph.astream(initial_state, run_config):
                yield state

            if config.require_human_approval:
                snapshot = await graph.aget_state(run_config)
                if snapshot and snapshot.next:
                    yield {
                        "__interrupt__": {
                            "awaiting_human_approval": True,
                            "next_nodes": list(snapshot.next),
                            "current_turn": snapshot.values.get("current_turn", 1),
                            "critiques_collection": snapshot.values.get("critiques_collection"),
                        }
                    }
    finally:
        if langfuse_handler is not None and hasattr(langfuse_handler, "flush"):
            try:
                langfuse_handler.flush()
            except Exception as e:
                logger.debug(f"Failed to flush Langfuse handler: {e}")


async def resume_masc_workflow(
    thread_id: str,
    critiques_override: Optional[CritiquesCollection] = None,
    approve: bool = True,
) -> AsyncGenerator[dict, None]:
    """
    Resumes a paused Human-in-the-Loop MASC workflow from its checkpointed state,
    optionally replacing the critiques_collection with human-curated critiques.
    """
    run_config: dict = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": settings.max_recursion_depth,
    }

    async with get_checkpointer() as checkpointer:
        probe_graph = build_masc_graph(checkpointer=checkpointer)
        snapshot = await probe_graph.aget_state(run_config)
        if not snapshot or not snapshot.values:
            raise ValueError(f"Thread '{thread_id}' not found.")

        if not approve:
            current_artifact = snapshot.values.get("current_artifact")
            yield {
                "aborted": {
                    "status": "rejected_by_human_reviewer",
                    "current_turn": snapshot.values.get("current_turn", 1),
                    "current_artifact": current_artifact,
                }
            }
            return

        if critiques_override is not None:
            await probe_graph.aupdate_state(
                run_config,
                {"critiques_collection": critiques_override},
            )

        # Resume without interrupt_before so the synthesis node executes cleanly
        resume_graph = build_masc_graph(checkpointer=checkpointer, interrupt_for_approval=False)
        async for state in resume_graph.astream(None, run_config):
            yield state
