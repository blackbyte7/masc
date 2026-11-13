import os

# --- Hermetic test environment -------------------------------------------------------------
# These MUST be set before any `src.*` import: `src.config.settings` instantiates MASCSettings at
# import time, and would otherwise read the developer's real `.env` (real DATABASE_URL, real
# provider keys, Langfuse credentials, MASC_API_KEY, ...).
os.environ["MASC_ENV_FILE"] = os.path.join(os.path.dirname(__file__), ".env.does-not-exist")
os.environ["DATABASE_URL"] = "memory://"
os.environ["OPENAI_API_KEY"] = "test-key-openai"
os.environ["ANTHROPIC_API_KEY"] = "test-key-anthropic"
os.environ["GOOGLE_API_KEY"] = "test-key-google"
for _var in ("MASC_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "TRUST_PROXY_HEADERS"):
    os.environ.pop(_var, None)

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src.core.credentials import vault  # noqa: E402
from src.core.state import LLMConfig, MASCConfig, PersonaConfig  # noqa: E402
from src.entrypoints.api_server import app  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_credential_vault():
    vault.clear()
    yield
    vault.clear()


@pytest.fixture
def test_client():
    return TestClient(app)


@pytest.fixture
def default_llm_config():
    return LLMConfig(
        provider="OpenAI",
        api_key="test-key-openai",
        model_name="gpt-5.2-pro",
        temperature=0.0,
    )


@pytest.fixture
def base_masc_config(default_llm_config):
    return MASCConfig(
        proposer=PersonaConfig(persona_name="Proposer", llm_config=default_llm_config),
        synthesizer=PersonaConfig(persona_name="Synthesizer", llm_config=default_llm_config),
        adversaries=[PersonaConfig(persona_name="DA", llm_config=default_llm_config)],
        synthesis_protocol="Sequential Refinement",
        max_turns=1,
    )
