from typing import List, Literal, Optional
from urllib.parse import urlparse

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MASCSettings(BaseSettings):
    # Core Limits
    max_adversaries: int = Field(default=5, description="Maximum allowed concurrent adversaries.")
    max_turns_limit: int = Field(default=10, description="Maximum allowed critique-synthesis cycles.")
    max_recursion_depth: int = Field(default=25, description="LangGraph recursion limit to prevent infinite loops.")
    max_task_length: int = Field(default=100000, description="Maximum character length for task descriptions.")

    # LLM Defaults
    default_provider: Literal["OpenAI", "Anthropic", "Google", "Ollama"] = Field(default="OpenAI")
    default_model: str = Field(default="gpt-5.2-pro")
    default_temperature: float = Field(default=0.1, description="Default temperature for agent nodes.")
    default_max_tokens: int = Field(default=4096, description="Budget limiter for generation steps.")

    # Resiliency & Streaming
    llm_retry_attempts: int = Field(default=3, description="Number of retries for API failures.")
    sse_heartbeat_interval: float = Field(
        default=15.0, description="Interval in seconds between SSE keep-alive comments."
    )

    # Security & API Configuration
    masc_api_key: Optional[str] = Field(
        default=None, description="Optional Bearer/API key to protect the FastAPI and Gradio servers."
    )
    cors_origins: List[str] = Field(
        default=["*"], description="Allowed CORS origins for the FastAPI server."
    )
    allowed_base_urls: List[str] = Field(
        default=[
            "http://localhost:11434",
            "http://127.0.0.1:11434",
            "http://host.docker.internal:11434",
        ],
        description="Allowlisted custom base URLs to prevent SSRF and API key exfiltration.",
    )
    allow_custom_base_urls: bool = Field(
        default=False,
        description="If True, permits non-allowlisted http/https base_urls (still blocks link-local metadata IPs).",
    )

    # Checkpointing / Persistence
    database_url: str = Field(
        default="memory://",
        description="URI for state persistence (use postgresql://... for persistent storage, or memory://).",
    )
    max_in_memory_threads: int = Field(
        default=100,
        ge=1,
        description="Maximum number of threads retained in the in-memory checkpointer to prevent memory leaks.",
    )

    # Observability (Optional Langfuse Integration)
    langfuse_secret_key: Optional[str] = Field(default=None, description="Langfuse Secret Key")
    langfuse_public_key: Optional[str] = Field(default=None, description="Langfuse Public Key")
    langfuse_host: str = Field(default="https://cloud.langfuse.com", description="Langfuse Host")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = MASCSettings()

_BLOCKED_HOSTS = {"169.254.169.254", "metadata.google.internal", "100.100.100.200"}


def validate_base_url(base_url: Optional[str], sys_settings: MASCSettings = settings) -> Optional[str]:
    """
    Validates a user-supplied base_url against SSRF and credential exfiltration risks.
    Returns the normalized URL or None if empty.
    """
    if not base_url or not base_url.strip():
        return None

    normalized = base_url.strip().rstrip("/")
    parsed = urlparse(normalized)

    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(f"Invalid base_url '{base_url}': must be a valid http or https URL.")

    hostname = (parsed.hostname or "").lower()
    if hostname in _BLOCKED_HOSTS or hostname.startswith("169.254."):
        raise ValueError(f"Disallowed base_url host '{hostname}'.")

    allowed_normalized = {u.strip().rstrip("/") for u in sys_settings.allowed_base_urls}
    if not sys_settings.allow_custom_base_urls and normalized not in allowed_normalized:
        raise ValueError(
            f"Disallowed base_url '{normalized}'. Add it to ALLOWED_BASE_URLS or enable ALLOW_CUSTOM_BASE_URLS."
        )

    return normalized
