import asyncio
import logging
import traceback
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional, Literal

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from psycopg_pool import AsyncConnectionPool

from src.config.settings import settings
from src.core.personas import ADVERSARY_PERSONAS
from src.core.state import MASCState, MASCConfig, PersonaConfig, Artifact, CritiquesCollection, Critique

logger = logging.getLogger(__name__)

_pg_pool: Optional[AsyncConnectionPool] = None


@asynccontextmanager
async def get_checkpointer():
    """Provides a managed checkpointer using a global connection pool."""
    global _pg_pool

    if settings.database_url.startswith("postgres"):
        if _pg_pool is None:
            logger.info("Initializing global PostgreSQL connection pool...")
            _pg_pool = AsyncConnectionPool(
                conninfo=settings.database_url,
                max_size=20,
                kwargs={"autocommit": True, "prepare_threshold": 0}
            )
            # Wait for the pool to connect successfully
            await _pg_pool.open()

        checkpointer = AsyncPostgresSaver(_pg_pool)
        await checkpointer.setup()
        yield checkpointer
    else:
        logger.warning("No Postgres URL found. Using ephemeral in-memory checkpointer.")
        yield MemorySaver()


class MASCNodes:
    def __init__(self, sys_settings):
        self.settings = sys_settings

    def _get_llm(self, config: PersonaConfig):
        provider = config.llm_config.provider

        if provider == "Ollama":
            try:
                from langchain_ollama import ChatOllama
            except ImportError:
                raise ImportError("Please run: pip install langchain-ollama")

            return ChatOllama(
                model=config.llm_config.model_name,
                base_url=config.llm_config.base_url or "http://localhost:11434",
                temperature=config.llm_config.temperature,
            )

        provider_mapping = {"OpenAI": "openai", "Anthropic": "anthropic", "Google": "google_genai"}
        provider_str = provider_mapping.get(provider)
        if not provider_str:
            raise ValueError(f"Unsupported LLM Provider: {provider}")

        retry_config = {"stop_after_attempt": self.settings.llm_retry_attempts, "wait_exponential_jitter": False}

        return init_chat_model(
            model=config.llm_config.model_name,
            model_provider=provider_str,
            temperature=config.llm_config.temperature,
            max_tokens=config.llm_config.max_tokens,
            api_key=config.llm_config.api_key or None,
            base_url=config.llm_config.base_url or None
        ).with_retry(**retry_config)

    async def propose(self, state: MASCState) -> dict:
        logger.info("Entering Proposer Node")
        config = state['config'].proposer
        llm = self._get_llm(config)

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             f"You are an expert {config.persona_name}. Your task is to generate a comprehensive, well-structured, and "
             f"well-supported initial response to the user's query. Build the strongest possible initial case."),
            ("human", "{task_description}")
        ])

        chain = prompt | llm
        response = await chain.ainvoke({"task_description": state['task_description']})

        return {
            "current_artifact": Artifact(
                version="1.0_proposal",
                content=response.content,
                history=[f"Initial proposal by {config.persona_name} using {config.llm_config.model_name}"]
            ),
            "current_turn": 1
        }

    async def adversarial_analysis(self, state: MASCState) -> dict:
        logger.info(f"Entering Adversarial Analysis Node (Turn {state.get('current_turn', 1)})")
        current_content = state['current_artifact'].content
        adversary_configs = state['config'].adversaries

        critique_tasks = [
            self._run_single_adversary(adv_config, adv_config.persona_name, current_content)
            for adv_config in adversary_configs
            if adv_config.persona_name in ADVERSARY_PERSONAS
        ]

        results = await asyncio.gather(*critique_tasks, return_exceptions=True)

        valid_critiques = []
        for res in results:
            if isinstance(res, Critique):
                valid_critiques.append(res)
            elif isinstance(res, Exception):
                logger.error(f"Adversary task failed with exception: {res}\n{traceback.format_exc()}")

        return {"critiques_collection": CritiquesCollection(critiques=valid_critiques)}

    async def _run_single_adversary(self, adv_config: PersonaConfig, name: str, content: str) -> Optional[Critique]:
        try:
            llm = self._get_llm(adv_config)
            persona = ADVERSARY_PERSONAS[name]
            structured_llm = llm.with_structured_output(Critique)

            prompt = ChatPromptTemplate.from_messages([
                ("system", persona["system_prompt"]),
                ("human", "{artifact_content}")
            ])

            chain = prompt | structured_llm
            logger.info(f"Running adversary: {name}")

            result = await chain.ainvoke({"artifact_content": content})
            logger.info(f"Successfully generated critique from {name}")
            return result
        except Exception as e:
            logger.error(f"Adversary '{name}' failed. Error: {e}")
            raise e

    async def triage_critiques(self, state: MASCState) -> dict:
        logger.info("Entering Triage Node (for Sequential Refinement)")
        config = state['config'].synthesizer
        llm = self._get_llm(config)

        critiques_json = state['critiques_collection'].model_dump_json(indent=2)
        task_description = state['task_description']

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a Triage Planner. Your task is to analyze the source, type, and severity of each critique. "
             "Sort the critiques into a prioritized processing queue to ensure the most critical issues are handled first. "
             "The default order should be: ANTAGONISTIC critiques first, then CRITICAL, HIGH, MEDIUM, and LOW severity CONSTRUCTIVE critiques. "
             "Your output must be the complete, reordered list of the original critique objects."),
            ("human", "Task Description: {task_description}\n\nUnordered Critiques:\n\n{critiques_json}")
        ])

        triage_chain = prompt | llm.with_structured_output(CritiquesCollection)
        logger.info("Re-ordering critiques for optimal refinement...")

        sorted_critiques_collection = await triage_chain.ainvoke({
            "task_description": task_description,
            "critiques_json": critiques_json
        })

        logger.info("Triage complete. New processing order established.")
        return {"critiques_collection": sorted_critiques_collection}

    async def synthesize_sequential(self, state: MASCState) -> dict:
        logger.info("Entering Synthesizer Node (Sequential Refinement)")
        config = state['config'].synthesizer
        llm = self._get_llm(config)

        current_artifact = state['current_artifact'].model_copy(deep=True)
        history_log = current_artifact.history.copy()
        current_turn = state.get("current_turn", 1)

        for critique in state['critiques_collection'].critiques:
            try:
                if critique.critique_type == 'ANTAGONISTIC':
                    logger.info(f"Refuting antagonistic critique from {critique.source_adversary}")
                    refutation_prompt = ChatPromptTemplate.from_messages([
                        ("system",
                         "You are a Master Synthesizer. Your task is to harden an original proposal against this cynical, "
                         "antagonistic argument. Identify the core fallacious principle in the critique and revise the "
                         "proposal to explicitly defend against it. Add justifications or foundational principles"
                         " that make the artifact's core ideology more robust. Do not integrate the substance of the "
                         "antagonistic feedback; your goal is to build a defense against it."),
                        ("human", "Original Proposal:\n\n{proposal}\n\nAntagonistic Critique:\n\n{ant_critique}")
                    ])
                    refutation_chain = refutation_prompt | llm
                    response = await refutation_chain.ainvoke({
                        "proposal": current_artifact.content,
                        "ant_critique": critique.model_dump_json()
                    })
                    current_artifact.content = response.content
                    history_log.append(f"[Turn {current_turn}] Hardened against: {critique.source_adversary}")

                elif critique.critique_type == 'CONSTRUCTIVE':
                    for issue in critique.payload:
                        logger.info(
                            f"Addressing constructive issue from {critique.source_adversary}: {issue.get('description', 'N/A')}")
                        revision_prompt = ChatPromptTemplate.from_messages([
                            ("system",
                             "You are a master editor. Your task is to revise the following draft to address ONLY the specific critique provided. "
                             "Your revision should be targeted and minimal, but comprehensive for the issue at hand. Do not change other parts of the draft."),
                            ("human",
                             "Current Draft:\n\n{draft}\n\nCritique to address:\n{issue_description}\n\nSuggested Recommendation:\n{recommendation}")
                        ])
                        revision_chain = revision_prompt | llm
                        response = await revision_chain.ainvoke({
                            "draft": current_artifact.content,
                            "issue_description": issue.get('description'),
                            "recommendation": issue.get('recommendation')
                        })
                        current_artifact.content = response.content
                        history_log.append(
                            f"[Turn {current_turn}] Fixed critique: '{issue.get('description')}' from {critique.source_adversary}.")
            except Exception as e:
                logger.error(f"Failed to process critique from {critique.source_adversary}. Error: {e}")
                history_log.append(
                    f"[Turn {current_turn}] ERROR: Failed to apply fix for critique from {critique.source_adversary}.")

        final_artifact = Artifact(version=f"{current_turn + 1}.0_sequential", content=current_artifact.content,
                                  history=history_log)
        return {"current_artifact": final_artifact, "current_turn": current_turn + 1}

    async def synthesize_architect(self, state: MASCState) -> dict:
        logger.info("Entering Synthesizer Node (Architect Protocol)")
        config = state['config'].synthesizer
        llm = self._get_llm(config)

        current_artifact = state['current_artifact']
        critiques_collection = state['critiques_collection']
        current_turn = state.get("current_turn", 1)

        history_log = current_artifact.history.copy()
        history_log.append(
            f"[Turn {current_turn}] Architect holistic review of {len(critiques_collection.critiques)} critiques.")

        architect_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a Chief AI Architect. You are presented with an initial proposal and a collection of adversarial critiques. "
             "Your task is to analyze all critiques simultaneously, resolve conflicting advice, and rewrite the proposal from the ground up "
             "to create a single, unified, and hardened final document. Ensure all valid concerns are integrated organically, "
             "rather than patched on sequentially."),
            ("human", "Initial Proposal:\n{proposal}\n\nAdversarial Critiques:\n{critiques}")
        ])

        architect_chain = architect_prompt | llm
        response = await architect_chain.ainvoke({
            "proposal": current_artifact.content,
            "critiques": critiques_collection.model_dump_json(indent=2)
        })

        history_log.append(f"[Turn {current_turn}] Architect unified critiques into a hardened artifact.")
        final_artifact = Artifact(version=f"{current_turn + 1}.0_architect", content=response.content, history=history_log)

        return {"current_artifact": final_artifact, "current_turn": current_turn + 1}


def should_route_to_synthesis(state: MASCState) -> Literal["triage_critiques", "synthesize_architect"]:
    protocol = state['config'].synthesis_protocol
    logger.info(f"Routing to synthesis protocol: {protocol}")
    if protocol == 'Architect':
        return "synthesize_architect"
    return "triage_critiques"


def should_continue_loop(state: MASCState) -> Literal["adversarial_analysis", "__end__"]:
    if state["current_turn"] <= state["config"].max_turns:
        logger.info(f"Looping back for Turn {state['current_turn']} / {state['config'].max_turns}")
        return "adversarial_analysis"
    logger.info("Max turns reached. Ending workflow.")
    return "__end__"


def build_masc_graph(checkpointer) -> CompiledStateGraph:
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
            "synthesize_architect": "synthesize_architect"
        }
    )

    graph_builder.add_edge("triage_critiques", "synthesize_sequential")

    # Cyclic routing
    graph_builder.add_conditional_edges("synthesize_sequential", should_continue_loop,
                                        {"adversarial_analysis": "adversarial_analysis", "__end__": END})
    graph_builder.add_conditional_edges("synthesize_architect", should_continue_loop,
                                        {"adversarial_analysis": "adversarial_analysis", "__end__": END})

    return graph_builder.compile(checkpointer=checkpointer)


async def execute_masc_workflow(task: str, config: MASCConfig, thread_id: str) -> AsyncGenerator[dict, None]:
    """
    Executes the workflow. The checkpointer is managed safely via the global pool.
    """
    initial_state = {"task_description": task, "config": config, "current_turn": 1}
    run_config = {"configurable": {"thread_id": thread_id}, "recursion_limit": settings.max_recursion_depth}

    if settings.langfuse_public_key and settings.langfuse_secret_key:
        try:
            from langfuse.callback import CallbackHandler
            langfuse_handler = CallbackHandler(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host
            )
            run_config["callbacks"] = [langfuse_handler]
            logger.info("Langfuse observability enabled for this execution.")
        except ImportError:
            logger.warning("Langfuse keys found, but langfuse package is not installed. Run 'pip install langfuse'.")

    async with get_checkpointer() as checkpointer:
        graph = build_masc_graph(checkpointer=checkpointer)
        async for state in graph.astream(initial_state, run_config):
            yield state
