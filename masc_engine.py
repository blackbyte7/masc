import logging
import asyncio
from typing import List, Literal, Optional, TypedDict, Annotated
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langgraph.graph import END, StateGraph
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class LLMConfig(BaseModel):
    provider: Literal["OpenAI", "Anthropic", "Google"]
    api_key: str = Field(..., repr=False)
    model_name: str
    temperature: float = 0.1

    model_config = {"arbitrary_types_allowed": True}


class PersonaConfig(BaseModel):
    persona_name: str
    llm_config: LLMConfig

    model_config = {"arbitrary_types_allowed": True}


class MASCConfig(BaseModel):
    proposer: PersonaConfig
    synthesizer: PersonaConfig
    adversaries: List[PersonaConfig]
    synthesis_protocol: Literal['Sequential Refinement', 'Architect'] = 'Sequential Refinement'

    model_config = {"arbitrary_types_allowed": True}


class Artifact(BaseModel):
    version: str
    content: str
    history: list = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}


class Critique(BaseModel):
    source_adversary: str = Field(description="The name of the adversary generating this critique.")
    critique_type: Literal["CONSTRUCTIVE", "ANTAGONISTIC"] = Field(description="The nature of the critique.")
    payload: Annotated[List[dict], Field(min_items=1)] = Field(description="A list of specific issues identified.")

    model_config = {"arbitrary_types_allowed": True}


class CritiquesCollection(BaseModel):
    critiques: List[Critique]

    model_config = {"arbitrary_types_allowed": True}


class MASCState(TypedDict):
    task_description: str
    config: MASCConfig
    v1_proposal: Artifact
    critiques_collection: Optional[CritiquesCollection]
    final_synthesis: Optional[Artifact]


ADVERSARY_PERSONAS = {
    "UncertaintyQuantifier": {
        "critique_type": "CONSTRUCTIVE",
        "system_prompt": "Assume the role of The Uncertainty Quantifier. Your sole function is to perform a rigorous "
                         "epistemic audit of the provided text. Your analysis checklist includes: Unsubstantiated Claims,"
                         " Ambiguous Terminology, Unstated Assumptions, Overconfident Extrapolation, and Neglect of Contested Knowledge."
                         " Critique the following artifact. Provide your output as a list of dictionary objects with 'severity', 'description',"
                         " and 'recommendation' keys."
    },
    "ContextualChallenger": {
        "critique_type": "CONSTRUCTIVE",
        "system_prompt": "Assume the role of The Contextual Challenger. Your function is to analyze the provided text"
                         " for its contextual blind spots. Your analysis checklist includes: Stakeholder Omissions,"
                         " Ethical and Moral Blind Spots, Cross-Domain Myopia, Implementation and Practicality Barriers,"
                         " and Potential for Misuse. Critique the following artifact. Provide your output as a list of"
                         " dictionary objects with 'severity', 'description', and 'recommendation' keys."
    },
    "DA": {
        "critique_type": "ANTAGONISTIC",
        "system_prompt": "Assume the role of The Devil's Advocate. Your function is not to provide constructive feedback. "
                         "Your task is to be a pure antagonist. Identify the central thesis of the provided text and "
                         "formulate the strongest possible counter-argument, even if that argument is specious or contrarian. "
                         "Your goal is to force a defense of the proposal's most fundamental assumptions. Do not offer solutions. "
                         "Your output is designed to be refuted, not integrated. Critique the following artifact. "
                         "Provide your output as a list of dictionary objects with 'severity' (always 'HIGH'), 'description' "
                         "(the counter-argument), and 'recommendation' (always 'Refute this ideological challenge.')."
    },
    "CodeAuditor": {
        "critique_type": "CONSTRUCTIVE",
        "system_prompt": "Assume the role of an expert Code Auditor. Your function is to perform a security and quality "
                         "audit of the provided source code. Your analysis checklist includes: Security Vulnerabilities "
                         "(SQL Injection, XSS, etc.), Performance Bottlenecks, Readability, Maintainability, Error Handling,"
                         " and adherence to Best Practices. Critique the following code artifact. Provide your output as a list of "
                         "dictionary objects with 'severity' ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW'), 'description', and 'recommendation' keys."
    }
}


class MASCNodes:
    def _get_llm(self, llm_config: LLMConfig) -> Runnable:
        retry_config = {"stop_after_attempt": 3, "wait_exponential_jitter": False}
        if llm_config.provider == "OpenAI":
            return ChatOpenAI(model=llm_config.model_name, temperature=llm_config.temperature,
                              api_key=llm_config.api_key).with_retry(**retry_config)
        if llm_config.provider == "Anthropic":
            return ChatAnthropic(model=llm_config.model_name, temperature=llm_config.temperature,
                                 api_key=llm_config.api_key).with_retry(**retry_config)
        if llm_config.provider == "Google":
            return ChatGoogleGenerativeAI(model=llm_config.model_name, temperature=llm_config.temperature,
                                          api_key=llm_config.api_key).with_retry(**retry_config)
        raise ValueError(f"Unsupported LLM Provider: {llm_config.provider}")

    async def propose(self, state: MASCState) -> dict:
        logging.info("Entering Proposer Node")
        config = state['config'].proposer
        llm = self._get_llm(config.llm_config)
        prompt = ChatPromptTemplate.from_messages([("system",
                                                    f"You are an expert {config.persona_name}. Your task is to generate a comprehensive, well-structured, and well-supported initial response to the user's query. Build the strongest possible initial case."),
                                                   ("human", "{task_description}")])
        chain = prompt | llm
        response = await chain.ainvoke({"task_description": state['task_description']})
        return {"v1_proposal": Artifact(version="1.0_proposal", content=response.content, history=[
            f"Initial proposal by {config.persona_name} using {config.llm_config.model_name}"])}

    async def adversarial_analysis(self, state: MASCState) -> dict:
        logging.info("Entering Adversarial Analysis Node")
        v1_proposal_content = state['v1_proposal'].content
        adversary_configs = state['config'].adversaries

        critique_tasks = []
        for adv_config in adversary_configs:
            name = adv_config.persona_name
            if name in ADVERSARY_PERSONAS:
                critique_tasks.append(
                    self._run_single_adversary(adv_config, name, v1_proposal_content)
                )

        results = await asyncio.gather(*critique_tasks)
        valid_critiques = [c for c in results if c is not None]

        return {"critiques_collection": CritiquesCollection(critiques=valid_critiques)}

    async def _run_single_adversary(self, adv_config: PersonaConfig, name: str, content: str) -> Optional[Critique]:
        try:
            llm = self._get_llm(adv_config.llm_config)
            persona = ADVERSARY_PERSONAS[name]

            structured_llm = llm.with_structured_output(Critique)

            prompt = ChatPromptTemplate.from_messages([
                ("system", persona["system_prompt"]),
                ("human", "{artifact_content}")
            ])

            chain = prompt | structured_llm

            logging.info(f"Running adversary: {name}")
            result = await chain.ainvoke({"artifact_content": content})
            logging.info(f"Successfully generated critique from {name}")
            return result

        except Exception as e:
            logging.error(f"Adversary '{name}' failed. Error: {e}")
            return None

    async def triage_critiques(self, state: MASCState) -> dict:
        logging.info("Entering Triage Node (for Sequential Refinement)")
        config = state['config'].synthesizer
        llm = self._get_llm(config.llm_config)
        critiques_json = state['critiques_collection'].json(indent=2)
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
        logging.info("Re-ordering critiques for optimal refinement...")
        sorted_critiques_collection = await triage_chain.ainvoke(
            {"task_description": task_description,
             "critiques_json": critiques_json
             })

        logging.info("Triage complete. New processing order established.")
        return {"critiques_collection": sorted_critiques_collection}

    async def synthesize_sequential(self, state: MASCState) -> dict:
        logging.info("Entering Synthesizer Node (Sequential Refinement)")
        config = state['config'].synthesizer
        llm = self._get_llm(config.llm_config)
        v1_proposal = state['v1_proposal']
        critiques_collection = state['critiques_collection']
        current_artifact = v1_proposal
        history_log = v1_proposal.history.copy()

        for critique in critiques_collection.critiques:
            try:
                if critique.critique_type == 'ANTAGONISTIC':
                    logging.info(f"Refuting antagonistic critique from {critique.source_adversary}")
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
                    response = await refutation_chain.ainvoke(
                        {"proposal": current_artifact.content, "ant_critique": critique.json()})
                    current_artifact.content = response.content
                    history_log.append(f"Hardened proposal against critique from: {critique.source_adversary}")

                elif critique.critique_type == 'CONSTRUCTIVE':
                    for issue in critique.payload:
                        logging.info(
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
                            f"Applied fix for critique: '{issue.get('description')}' from {critique.source_adversary}.")
            except Exception as e:
                logging.error(
                    f"Failed to process critique from {critique.source_adversary}. Error: {e}. Skipping to next critique.")
                history_log.append(
                    f"ERROR: Failed to apply fix for critique from {critique.source_adversary}. See logs for details.")

        final_artifact = Artifact(version="2.0_final_sequential", content=current_artifact.content, history=history_log)
        return {"final_synthesis": final_artifact}

    async def synthesize_architect(self, state: MASCState) -> dict:
        logging.info("Entering Synthesizer Node (Architect Protocol - Placeholder)")
        v1_proposal = state['v1_proposal']
        critiques_collection = state['critiques_collection']

        history_log = v1_proposal.history.copy()
        history_log.append("Architect synthesis protocol selected. [This protocol is not fully implemented].")
        history_log.append(f"Received {len(critiques_collection.critiques)} critiques for holistic review.")

        # VIP, in a real implementation, this would involve clustering, conflict resolution etc.
        final_content = (f"--- ARCHITECT PROTOCOL OUTPUT (PLACEHOLDER) ---\n\n"
                         f"The 'Architect' synthesis protocol was activated. This advanced protocol would normally perform "
                         f"holistic critique clustering, conflict resolution, and a single, unified revision.\n\n"
                         f"As a placeholder, here is the original proposal that would have been refined:\n\n"
                         f"{v1_proposal.content}")

        final_artifact = Artifact(version="2.0_final_architect", content=final_content, history=history_log)
        return {"final_synthesis": final_artifact}


def should_route_to_synthesis(state: MASCState) -> Literal["triage", "architect"]:
    protocol = state['config'].synthesis_protocol
    logging.info(f"Routing to synthesis protocol: {protocol}")
    if protocol == 'Architect':
        return "architect"
    return "triage"


def create_masc_t_graph():
    """Builds the MASC-T graph with conditional routing for synthesis protocols."""
    nodes = MASCNodes()
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
            "triage": "triage_critiques",
            "architect": "synthesize_architect"
        }
    )

    graph_builder.add_edge("triage_critiques", "synthesize_sequential")
    graph_builder.add_edge("synthesize_sequential", END)
    graph_builder.add_edge("synthesize_architect", END)

    return graph_builder.compile()
