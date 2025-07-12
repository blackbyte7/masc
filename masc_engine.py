import logging
from typing import List, Literal, Optional, TypedDict
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, conlist
from langchain_core.runnables import RunnableParallel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# MASC-T Data & Configuration Structures
class LLMConfig(BaseModel):
    provider: Literal["OpenAI", "Anthropic", "Google"]
    api_key: str = Field(..., repr=False)
    model_name: str
    temperature: float = 0.1


class PersonaConfig(BaseModel):
    persona_name: str
    llm_config: LLMConfig


class MASCConfig(BaseModel):
    proposer: PersonaConfig
    synthesizer: PersonaConfig
    adversaries: List[PersonaConfig]


class Artifact(BaseModel):
    version: str
    content: str
    history: list = Field(default_factory=list)


class Critique(BaseModel):
    source_adversary: str
    critique_type: Literal["CONSTRUCTIVE", "ANTAGONISTIC"]
    payload: conlist(dict, min_items=1)


class CritiquesCollection(BaseModel):
    critiques: List[Critique]


class MASCState(TypedDict):
    task_description: str
    config: MASCConfig
    v1_proposal: Artifact
    critiques_collection: Optional[CritiquesCollection]
    final_synthesis: Optional[Artifact]


# Modular Adversary Library (Full Prompts Included)
ADVERSARY_PERSONAS = {
    "UncertaintyQuantifier": {
        "critique_type": "CONSTRUCTIVE",
        "system_prompt": "Assume the role of The Uncertainty Quantifier. Your sole function is to perform a rigorous "
                         "epistemic audit of the provided text. Your analysis checklist includes: Unsubstantiated Claims, "
                         "Ambiguous Terminology, Unstated Assumptions, Overconfident Extrapolation, and Neglect of Contested Knowledge. "
                         "Critique the following artifact. Provide your output as a list of dictionary objects with "
                         "'severity', 'description', and 'recommendation' keys."
    },
    "ContextualChallenger": {
        "critique_type": "CONSTRUCTIVE",
        "system_prompt": "Assume the role of The Contextual Challenger. Your function is to analyze the provided text "
                         "for its contextual blind spots. Your analysis checklist includes: Stakeholder Omissions, "
                         "Ethical and Moral Blind Spots, Cross-Domain Myopia, Implementation and Practicality Barriers,"
                         " and Potential for Misuse. Critique the following artifact. Provide your output as a list of"
                         " dictionary objects with 'severity', 'description', and 'recommendation' keys."
    },
    "DA": {  # Devil's Advocate
        "critique_type": "ANTAGONISTIC",
        "system_prompt": "Assume the role of The Devil's Advocate. Your function is not to provide constructive "
                         "feedback. Your task is to be a pure antagonist. Identify the central thesis of the provided "
                         "text and formulate the strongest possible counter-argument, even if that argument is specious"
                         " or contrarian. Your goal is to force a defense of the proposal's most fundamental assumptions."
                         " Do not offer solutions. Your output is designed to be refuted, not integrated. "
                         "Critique the following artifact. Provide your output as a list of dictionary objects with "
                         "'severity' (always 'HIGH'), 'description' (the counter-argument), and 'recommendation' "
                         "(always 'Refute this ideological challenge.')."
    },
    "CodeAuditor": {
        "critique_type": "CONSTRUCTIVE",
        "system_prompt": "Assume the role of an expert Code Auditor. Your function is to perform a security and quality"
                         " audit of the provided source code. Your analysis checklist includes: Security Vulnerabilities"
                         " (SQL Injection, XSS, etc.), Performance Bottlenecks, Readability, Maintainability, Error Handling,"
                         " and adherence to Best Practices. Critique the following code artifact. Provide your output as a"
                         " list of dictionary objects with 'severity' ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW'), 'description', and 'recommendation' keys."
    }
}


# MASC-T Agentic Graph Nodes (Stateless & Config-Driven)
class MASCNodes:
    def _get_llm(self, llm_config: LLMConfig):
        if llm_config.provider == "OpenAI": return ChatOpenAI(model=llm_config.model_name,
                                                              temperature=llm_config.temperature,
                                                              api_key=llm_config.api_key)
        if llm_config.provider == "Anthropic": return ChatAnthropic(model=llm_config.model_name,
                                                                    temperature=llm_config.temperature,
                                                                    api_key=llm_config.api_key)
        if llm_config.provider == "Google": return ChatGoogleGenerativeAI(model=llm_config.model_name,
                                                                          temperature=llm_config.temperature,
                                                                          api_key=llm_config.api_key)
        raise ValueError(f"Unsupported LLM Provider: {llm_config.provider}")

    def propose(self, state: MASCState) -> MASCState:
        logging.info("Entering Proposer Node")
        config = state['config'].proposer
        llm = self._get_llm(config.llm_config)
        prompt = ChatPromptTemplate.from_messages([("system",
                                                    f"You are an expert {config.persona_name}. "
                                                    f"Your task is to generate a comprehensive, well-structured, and "
                                                    f"well-supported initial response to the user's query. Build the strongest possible initial case."),
                                                   ("human", "{task_description}")])
        chain = prompt | llm
        content = chain.invoke({"task_description": state['task_description']}).content
        return {"v1_proposal": Artifact(version="1.0_proposal", content=content, history=[
            f"Initial proposal by {config.persona_name} using {config.llm_config.model_name}"])}

    def adversarial_analysis(self, state: MASCState) -> MASCState:
        logging.info("Entering Adversarial Analysis Node")
        v1_proposal_content = state['v1_proposal'].content
        adversary_configs = state['config'].adversaries
        runnables = {}
        for adv_config in adversary_configs:
            name = adv_config.persona_name
            if name in ADVERSARY_PERSONAS:
                llm = self._get_llm(adv_config.llm_config)
                persona = ADVERSARY_PERSONAS[name]
                prompt = ChatPromptTemplate.from_messages(
                    [("system", persona["system_prompt"]), ("human", "{artifact_content}")])
                runnables[name] = prompt | llm.with_structured_output(Critique)

        results = RunnableParallel(**runnables).invoke({"artifact_content": v1_proposal_content})
        for name, result in results.items():
            result.source_adversary = name
            result.critique_type = ADVERSARY_PERSONAS[name]['critique_type']
        return {"critiques_collection": CritiquesCollection(critiques=list(results.values()))}

    def synthesize(self, state: MASCState) -> MASCState:
        logging.info("Entering Synthesizer Node")
        config = state['config'].synthesizer
        llm = self._get_llm(config.llm_config)

        v1_proposal = state['v1_proposal']
        critiques_collection = state['critiques_collection']

        # Step F1: Triage
        antagonistic_critiques = [c for c in critiques_collection.critiques if c.critique_type == 'ANTAGONISTIC']
        constructive_critiques = [c for c in critiques_collection.critiques if c.critique_type == 'CONSTRUCTIVE']

        # Start with the original proposal
        current_artifact = v1_proposal
        history_log = v1_proposal.history.copy()

        # Step F2: Antagonist Refutation
        if antagonistic_critiques:
            logging.info(f"Refuting {len(antagonistic_critiques)} antagonistic critiques.")
            refutation_prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "You are a Master Synthesizer. Your task is to harden an original proposal against these cynical, antagonistic arguments. Identify the core fallacious principles in the critiques and revise the proposal to explicitly defend against them. Add justifications, preambles, or foundational principles that make the artifact's core ideology more robust. Do not integrate the substance of the antagonistic feedback; your goal is to build a defense against it."),
                ("human", "Original Proposal:\n\n{proposal}\n\nAntagonistic Critiques:\n\n{ant_critiques}")
            ])
            refutation_chain = refutation_prompt | llm
            response = refutation_chain.invoke({
                "proposal": current_artifact.content,
                "ant_critiques": "\n".join([str(c.payload) for c in antagonistic_critiques])
            })
            current_artifact.content = response.content
            history_log.append(
                f"Hardened proposal against antagonistic critiques from: {[c.source_adversary for c in antagonistic_critiques]}")

        # Step F3: Metacognitive Integration Loop
        logging.info(f"Integrating {len(constructive_critiques)} constructive critiques.")
        rca_chain = ChatPromptTemplate.from_messages([
            ("system",
             "You are a metacognitive analyst. Your task is purely analytical. Given the following critique of an artifact, do not solve the problem. Instead, identify the flawed underlying principle or mental model that led to this flaw, and then propose a superior, corrected principle. Output a structured object containing {{flawed_principle, corrected_principle}}."),
            ("human", "Critique:\n\n{critique}")
        ]) | llm

        revision_chain = ChatPromptTemplate.from_messages([
            ("system",
             "You are a master editor. Your task is purely generative. Take the provided draft and systemically rewrite the relevant sections to apply the corrected_principle. Ensure the new principle is applied globally and consistently to fix not just the cited flaw, but all instances of that flawed thinking."),
            ("human",
             "Current Draft:\n\n{draft}\n\nCorrection Plan:\n\nFlawed Principle: {flawed_principle}\nCorrected Principle: {corrected_principle}")
        ]) | llm

        for critique in constructive_critiques:
            for issue in critique.payload:
                logging.info(f"Addressing issue from {critique.source_adversary}: {issue.get('description', 'N/A')}")
                try:
                    # F3a: Root Cause Analysis
                    rca_response = rca_chain.invoke({"critique": issue['description']}).content
                    flawed_principle = rca_response.split("flawed_principle:")[1].split("corrected_principle:")[
                        0].strip()
                    corrected_principle = rca_response.split("corrected_principle:")[1].strip()

                    # F3b: Revision Application
                    revision_response = revision_chain.invoke({
                        "draft": current_artifact.content,
                        "flawed_principle": flawed_principle,
                        "corrected_principle": corrected_principle
                    })

                    current_artifact.content = revision_response.content
                    history_log.append(
                        f"Applied principle '{corrected_principle}' to address '{issue['description']}'.")
                except Exception as e:
                    logging.error(f"Could not process critique issue: {issue}. Error: {e}")
                    history_log.append(f"Failed to apply principle for critique: '{issue['description']}'.")

        final_artifact = Artifact(
            version="2.0_final",
            content=current_artifact.content,
            history=history_log
        )
        return {"final_synthesis": final_artifact}


def create_masc_t_graph():
    """Builds the MASC-T graph. LLMs are instantiated inside the nodes."""
    nodes = MASCNodes()
    graph_builder = StateGraph(MASCState)
    graph_builder.add_node("propose", nodes.propose)
    graph_builder.add_node("adversarial_analysis", nodes.adversarial_analysis)
    graph_builder.add_node("synthesize", nodes.synthesize)
    graph_builder.set_entry_point("propose")
    graph_builder.add_edge("propose", "adversarial_analysis")
    graph_builder.add_edge("adversarial_analysis", "synthesize")
    graph_builder.add_edge("synthesize", END)
    return graph_builder.compile()
