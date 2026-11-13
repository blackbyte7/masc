"""Single source of truth for request-level MASCConfig validation shared by every entrypoint."""

from typing import Optional

from src.config.settings import MASCSettings, settings, validate_base_url
from src.core.personas import get_persona_registry, resolve_adversary_persona
from src.core.state import MASCConfig


class ConfigValidationError(ValueError):
    """Raised when a MASCConfig violates server-side limits or security policy."""


def validate_masc_config(config: MASCConfig, sys_settings: MASCSettings = settings) -> None:
    """
    Enforces server limits (adversary count, turns), persona resolvability, SSRF protection on
    ``base_url`` and the environment-key exfiltration guard. Raises ``ConfigValidationError``.
    """
    if len(config.adversaries) > sys_settings.max_adversaries:
        raise ConfigValidationError(f"Exceeded max adversaries ({sys_settings.max_adversaries}).")

    if config.max_turns > sys_settings.max_turns_limit:
        raise ConfigValidationError(f"Exceeded max turns limit ({sys_settings.max_turns_limit}).")

    if config.initial_artifact and len(config.initial_artifact) > sys_settings.max_task_length:
        raise ConfigValidationError(
            f"initial_artifact exceeds maximum length ({sys_settings.max_task_length} characters)."
        )

    registry = get_persona_registry()
    invalid_personas = []
    for adv in config.adversaries:
        try:
            resolve_adversary_persona(adv, registry)
        except ValueError:
            invalid_personas.append(adv.persona_name)
    if invalid_personas:
        raise ConfigValidationError(f"Unknown adversary persona(s): {invalid_personas}")

    for persona in (config.proposer, config.synthesizer, *config.adversaries):
        llm_cfg = persona.llm_config
        try:
            validated_url = validate_base_url(llm_cfg.base_url, sys_settings)
        except ValueError as exc:
            raise ConfigValidationError(str(exc)) from exc

        raw_key = (llm_cfg.api_key or "").strip()
        if validated_url and llm_cfg.provider != "Ollama" and raw_key in ("", "env"):
            raise ConfigValidationError(
                f"Explicit API key is required when specifying a custom base_url for provider '{llm_cfg.provider}'."
            )


def validate_task_inputs(
    task_description: str,
    initial_artifact: Optional[str] = None,
    sys_settings: MASCSettings = settings,
) -> None:
    """Validates free-text inputs that are not covered by the pydantic request models."""
    if not task_description or not task_description.strip():
        raise ConfigValidationError("task_description cannot be empty.")
    if len(task_description) > sys_settings.max_task_length:
        raise ConfigValidationError(
            f"task_description exceeds maximum length ({sys_settings.max_task_length} characters)."
        )
    if initial_artifact and len(initial_artifact) > sys_settings.max_task_length:
        raise ConfigValidationError(
            f"initial_artifact exceeds maximum length ({sys_settings.max_task_length} characters)."
        )
