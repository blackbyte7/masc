from typing import List, Literal, Optional, TypedDict, Annotated

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    provider: Literal["OpenAI", "Anthropic", "Google", "Ollama"]
    api_key: str = Field(default="", repr=False)
    base_url: Optional[str] = Field(default=None, description="Custom local URL (e.g., http://localhost:11434)")
    model_name: str
    temperature: float = 0.1
    max_tokens: int = Field(default=4096, description="Budget limiter to prevent runaway token usage.")


class PersonaConfig(BaseModel):
    persona_name: str
    llm_config: LLMConfig


class MASCConfig(BaseModel):
    proposer: PersonaConfig
    synthesizer: PersonaConfig
    adversaries: List[PersonaConfig]
    synthesis_protocol: Literal['Sequential Refinement', 'Architect'] = 'Sequential Refinement'
    max_turns: int = Field(default=1, ge=1, description="Number of critique-synthesis cycles to execute.")


class Artifact(BaseModel):
    version: str
    content: str
    history: list[str] = Field(default_factory=list)


class Critique(BaseModel):
    source_adversary: str = Field(description="The name of the adversary generating this critique.")
    critique_type: Literal["CONSTRUCTIVE", "ANTAGONISTIC"] = Field(description="The nature of the critique.")
    payload: Annotated[List[dict], Field(min_items=1)] = Field(description="A list of specific issues identified.")


class CritiquesCollection(BaseModel):
    critiques: List[Critique]


class MASCState(TypedDict):
    task_description: str
    config: MASCConfig
    current_artifact: Artifact
    critiques_collection: Optional[CritiquesCollection]
    current_turn: int
