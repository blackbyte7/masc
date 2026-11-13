import ipaddress
import os
from typing import List, Literal, Optional, Union
from urllib.parse import urlparse

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# SQLAlchemy-style driver suffixes that psycopg does not understand (e.g. "postgresql+psycopg://").
_PG_DRIVER_SUFFIXES = ("+psycopg", "+psycopg2", "+psycopg3", "+asyncpg")


class MASCSettings(BaseSettings):
    # Core Limits
    max_adversaries: int = Field(default=5, ge=1, description="Maximum allowed concurrent adversaries.")
    max_turns_limit: int = Field(default=10, ge=1, description="Maximum allowed critique-synthesis cycles.")
    max_recursion_depth: int = Field(default=25, description="LangGraph recursion limit to prevent infinite loops.")
    max_task_length: int = Field(default=100000, description="Maximum character length for task descriptions.")
    max_concurrent_llm_calls: int = Field(
        default=10,
        ge=1,
        description="Global concurrency semaphore limit for simultaneous outbound LLM calls.",
    )
    rate_limit_per_minute: int = Field(
        default=60,
        ge=0,
        description="Maximum execution requests per minute per client IP (set to 0 to disable rate limiting).",
    )

    # LLM Defaults
    default_provider: Literal["OpenAI", "Anthropic", "Google", "Ollama"] = Field(default="OpenAI")
    default_model: str = Field(default="gpt-5.2-pro")
    default_temperature: float = Field(default=0.1, description="Default temperature for agent nodes.")
    default_max_tokens: int = Field(default=4096, description="Budget limiter for generation steps.")

    # Resiliency & Streaming
    llm_retry_attempts: int = Field(default=3, ge=1, description="Number of retries for API failures.")
    sse_heartbeat_interval: float = Field(
        default=15.0, gt=0, description="Interval in seconds between SSE keep-alive comments."
    )

    # Security & API Configuration
    masc_api_key: Optional[str] = Field(
        default=None, description="Optional Bearer/API key to protect the FastAPI and Gradio servers."
    )
    cors_origins: List[str] = Field(default=["*"], description="Allowed CORS origins for the FastAPI server.")
    trust_proxy_headers: bool = Field(
        default=False,
        description=(
            "If True, the client IP used for rate limiting is taken from the first X-Forwarded-For entry. "
            "Only enable this when MASC runs behind a trusted reverse proxy that overwrites the header."
        ),
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
    pg_pool_max_size: int = Field(
        default=20,
        ge=1,
        description="Maximum number of pooled PostgreSQL connections used by the checkpointer.",
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

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Root log level for all MASC entrypoints."
    )

    # MASC_ENV_FILE lets tests and deployments point at a different dotenv file (or a non-existent one).
    model_config = SettingsConfigDict(
        env_file=os.environ.get("MASC_ENV_FILE", ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    @field_validator("database_url")
    @classmethod
    def _normalize_database_url(cls, value: str) -> str:
        """Accepts SQLAlchemy-style URLs (postgresql+psycopg://) by stripping the driver suffix for psycopg."""
        value = value.strip()
        scheme, sep, rest = value.partition("://")
        if sep and scheme.lower().startswith("postgres"):
            lowered = scheme.lower()
            for suffix in _PG_DRIVER_SUFFIXES:
                if lowered.endswith(suffix):
                    scheme = scheme[: -len(suffix)]
                    break
            return f"{scheme}{sep}{rest}"
        return value

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper_log_level(cls, value: object) -> object:
        return value.upper() if isinstance(value, str) else value

    @property
    def uses_postgres(self) -> bool:
        """True when the checkpointer should use PostgreSQL rather than the bounded in-memory saver."""
        return self.database_url.lower().startswith(("postgres://", "postgresql://"))


settings = MASCSettings()

_IPAddress = Union[ipaddress.IPv4Address, ipaddress.IPv6Address]
_BLOCKED_HOSTS = {"169.254.169.254", "metadata.google.internal", "metadata", "100.100.100.200", "fd00:ec2::254"}


def _parse_ip_host(hostname: str) -> Optional[_IPAddress]:
    """
    Parses IPv4/IPv6 literals, including the legacy integer/hex/octal IPv4 notations that
    HTTP clients still resolve (e.g. http://2852039166/ == http://169.254.169.254/).
    """
    try:
        return ipaddress.ip_address(hostname)
    except ValueError:
        pass
    try:
        if hostname.isdigit():
            return ipaddress.IPv4Address(int(hostname))
        if hostname.lower().startswith("0x"):
            return ipaddress.IPv4Address(int(hostname, 16))
    except (ValueError, ipaddress.AddressValueError):
        return None
    return None


def _is_forbidden_ip(ip: _IPAddress) -> bool:
    """Blocks cloud metadata / link-local / unspecified / multicast / reserved targets. Loopback stays allowed."""
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        ip = mapped
    return ip.is_link_local or ip.is_unspecified or ip.is_multicast or ip.is_reserved or str(ip) in _BLOCKED_HOSTS


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
    if parsed.username or parsed.password:
        raise ValueError("Invalid base_url: embedded credentials (user:pass@host) are not allowed.")

    hostname = (parsed.hostname or "").lower().rstrip(".")
    if not hostname:
        raise ValueError(f"Invalid base_url '{base_url}': missing hostname.")

    ip = _parse_ip_host(hostname)
    if hostname in _BLOCKED_HOSTS or hostname.startswith("169.254.") or (ip is not None and _is_forbidden_ip(ip)):
        raise ValueError(f"Disallowed base_url host '{hostname}'.")

    allowed_normalized = {u.strip().rstrip("/") for u in sys_settings.allowed_base_urls}
    if not sys_settings.allow_custom_base_urls and normalized not in allowed_normalized:
        raise ValueError(
            f"Disallowed base_url '{normalized}'. Add it to ALLOWED_BASE_URLS or enable ALLOW_CUSTOM_BASE_URLS."
        )

    return normalized
