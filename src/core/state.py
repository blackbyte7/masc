from typing import Annotated, Any, List, Literal, Optional, TypedDict

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    provider: Literal["OpenAI", "Anthropic", "Google", "Ollama"]
    api_key: str = Field(default="", repr=False)
    base_url: Optional[str] = Field(default=None, description="Custom local URL (e.g., http://localhost:11434)")
    model_name: str
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1, description="Budget limiter to prevent runaway token usage.")

    def redacted_dump(self) -> dict[str, Any]:
        """Returns a dictionary representation with the API key removed."""
        data = self.model_dump()
        data.pop("api_key", None)
        return data


class PersonaConfig(BaseModel):
    persona_name: str
    llm_config: LLMConfig


class MASCConfig(BaseModel):
    proposer: PersonaConfig
    synthesizer: PersonaConfig
    adversaries: Annotated[List[PersonaConfig], Field(min_length=1)]
    synthesis_protocol: Literal["Sequential Refinement", "Architect"] = "Sequential Refinement"
    max_turns: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Number of critique-synthesis cycles to execute.",
    )

    def redacted_dump(self) -> dict[str, Any]:
        """Returns a dictionary representation with all LLM API keys redacted."""
        data = self.model_dump()
        if "proposer" in data and "llm_config" in data["proposer"]:
            data["proposer"]["llm_config"].pop("api_key", None)
        if "synthesizer" in data and "llm_config" in data["synthesizer"]:
            data["synthesizer"]["llm_config"].pop("api_key", None)
        for adv in data.get("adversaries", []):
            if "llm_config" in adv:
                adv["llm_config"].pop("api_key", None)
        return data


class Artifact(BaseModel):
    version: str
    content: str
    history: List[str] = Field(default_factory=list)


class CritiqueItem(BaseModel):
    severity: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"] = Field(
        description="Severity of the identified issue ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW')."
    )
    description: str = Field(
        description="Detailed explanation of the vulnerability, blind spot, or counter-argument."
    )
    recommendation: str = Field(
        description="Concrete recommendation to resolve the issue or refute the challenge."
    )

    def get(self, key: str, default: Any = None) -> Any:
        """Provides dictionary-style `.get()` compatibility alongside attribute access."""
        return getattr(self, key, default)


class CritiqueLLMOutput(BaseModel):
    """Schema used exclusively for structured LLM generation by an adversary."""

    payload: List[CritiqueItem] = Field(
        default_factory=list,
        description="A list of specific issues identified during the audit. Return an empty list if no valid issues exist.",
    )


class Critique(BaseModel):
    source_adversary: str = Field(description="The name of the adversary generating this critique.")
    critique_type: Literal["CONSTRUCTIVE", "ANTAGONISTIC"] = Field(description="The nature of the critique.")
    payload: List[CritiqueItem] = Field(
        default_factory=list,
        description="A list of specific issues identified.",
    )


class CritiquesCollection(BaseModel):
    critiques: List[Critique] = Field(default_factory=list)


class MASCState(TypedDict):
    task_description: str
    config: MASCConfig
    current_artifact: Artifact
    critiques_collection: Optional[CritiquesCollection]
    current_turn: int
