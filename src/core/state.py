from typing import Any, List, Literal, Optional, TypedDict

from pydantic import BaseModel, Field, model_validator


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
    critique_type: Optional[Literal["CONSTRUCTIVE", "ANTAGONISTIC"]] = Field(
        default=None,
        description="Optional inline critique type for ephemeral/request-scoped adversary personas.",
    )
    system_prompt: Optional[str] = Field(
        default=None,
        max_length=20000,
        description="Optional inline system prompt for ephemeral/request-scoped adversary personas.",
    )

    @model_validator(mode="after")
    def validate_inline_persona_fields(self) -> "PersonaConfig":
        has_type = self.critique_type is not None
        has_prompt = bool(self.system_prompt and self.system_prompt.strip())
        if has_type != has_prompt:
            raise ValueError(
                f"Persona '{self.persona_name}': both 'critique_type' and a non-empty 'system_prompt' "
                "must be provided together when defining an inline persona."
            )
        return self


class MASCConfig(BaseModel):
    proposer: PersonaConfig
    synthesizer: PersonaConfig
    adversaries: List[PersonaConfig] = Field(
        default_factory=list,
        description="Configured adversary personas. Must contain at least 1 adversary unless auto_select_adversaries is True.",
    )
    auto_select_adversaries: bool = Field(
        default=False,
        description=(
            "Optional feature: if True, dynamically routes and selects the most relevant adversary personas "
            "from the persona registry based on the task description."
        ),
    )
    synthesis_protocol: Literal["Sequential Refinement", "Architect"] = "Sequential Refinement"
    max_turns: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Number of critique-synthesis cycles to execute.",
    )
    stop_on_max_severity: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW", "NONE"] = Field(
        default="LOW",
        description=(
            "Minimum critique severity required to continue looping. "
            "For example, if set to 'HIGH', the workflow stops early when only 'MEDIUM' or 'LOW' issues remain. "
            "Set to 'NONE' to disable severity-based early convergence."
        ),
    )
    batch_constructive_critiques: bool = Field(
        default=False,
        description=(
            "If True during Sequential Refinement, merges all triaged constructive critiques into a single "
            "unified synthesis pass after refuting antagonistic critiques."
        ),
    )
    enable_verification_tools: bool = Field(
        default=False,
        description=(
            "Optional feature: if True, runs deterministic static verification checks (AST syntax, secret scanning, "
            "dangerous function checks) and injects findings into adversary prompts."
        ),
    )
    require_human_approval: bool = Field(
        default=False,
        description=(
            "Optional feature: if True, pauses graph execution before synthesis so a human reviewer can inspect, "
            "approve, or edit the critiques via the resume endpoint."
        ),
    )
    response_schema: Optional[dict[str, Any]] = Field(
        default=None,
        description="Optional JSON Schema to enforce structured JSON output on the final synthesized artifact.",
    )
    max_total_tokens: Optional[int] = Field(
        default=None,
        ge=100,
        description="Optional cumulative token budget across the workflow run; halts looping if exceeded.",
    )
    initial_artifact: Optional[str] = Field(
        default=None,
        description="Optional pre-existing artifact to audit directly, skipping the initial Proposer node.",
    )

    @model_validator(mode="after")
    def validate_adversaries_or_auto_router(self) -> "MASCConfig":
        if not self.adversaries and not self.auto_select_adversaries:
            raise ValueError(
                "At least one adversary must be provided in 'adversaries' when 'auto_select_adversaries' is False."
            )
        return self

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


class PersonaRouterOutput(BaseModel):
    """Schema used by the optional Auto-Persona Router node to select relevant critics."""

    selected_personas: List[str] = Field(
        default_factory=list,
        description="List of 1 to 3 adversary persona names best suited to audit the given task.",
    )
    reasoning: str = Field(
        default="",
        description="Brief justification for why these personas were selected.",
    )


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


class TokenUsage(BaseModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    llm_calls: int = Field(default=0, ge=0)

    def add(self, input_tokens: int = 0, output_tokens: int = 0, total_tokens: int = 0, calls: int = 1) -> "TokenUsage":
        resolved_total = total_tokens if total_tokens > 0 else (input_tokens + output_tokens)
        return TokenUsage(
            input_tokens=self.input_tokens + max(0, input_tokens),
            output_tokens=self.output_tokens + max(0, output_tokens),
            total_tokens=self.total_tokens + max(0, resolved_total),
            llm_calls=self.llm_calls + max(0, calls),
        )


class MASCState(TypedDict, total=False):
    task_description: str
    config: MASCConfig
    current_artifact: Artifact
    critiques_collection: Optional[CritiquesCollection]
    current_turn: int
    token_usage: TokenUsage
