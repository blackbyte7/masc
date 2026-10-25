import os

import pytest
from fastapi.testclient import TestClient

from src.core.state import LLMConfig, MASCConfig, PersonaConfig
from src.entrypoints.api_server import app

# Set dummy environment variables for test runs
os.environ["OPENAI_API_KEY"] = "test-key-openai"
os.environ["ANTHROPIC_API_KEY"] = "test-key-anthropic"
os.environ["GOOGLE_API_KEY"] = "test-key-google"
os.environ["DATABASE_URL"] = "memory://"


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
