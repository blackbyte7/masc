from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MASCSettings(BaseSettings):
    # Core Limits
    max_adversaries: int = Field(default=5, description="Maximum allowed concurrent adversaries.")
    max_recursion_depth: int = Field(default=15, description="LangGraph recursion limit to prevent infinite loops.")

    # LLM Defaults
    default_provider: Literal["OpenAI", "Anthropic", "Google"] = Field(default="OpenAI")
    default_model: str = Field(default="gpt-4o")
    default_temperature: float = Field(default=0.1, description="Default temperature for agent nodes.")
    default_max_tokens: int = Field(default=4096, description="Budget limiter for generation steps.")

    # Resiliency
    llm_retry_attempts: int = Field(default=3, description="Number of retries for API failures.")

    # Checkpointing / Persistence (Postgres connection string for LangGraph checkpointer)
    database_url: str = Field(default="sqlite:///masc_memory.db", description="URI for state persistence.")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = MASCSettings()
