import pytest

from src.core.state import Artifact, MASCConfig
from src.entrypoints.cli import generate_template
from src.entrypoints.mcp_server import trigger_masc_audit
from src.utils.model_fetcher import FALLBACK_MODELS, fetch_available_models


def test_cli_generate_template_has_empty_api_key(tmp_path):
    template_file = tmp_path / "config.json"
    generate_template(str(template_file))

    assert template_file.exists()
    config = MASCConfig.model_validate_json(template_file.read_text(encoding="utf-8"))
    # Ensure template no longer sets literal "env" as the API key
    assert config.proposer.llm_config.api_key == ""
    assert config.synthesizer.llm_config.api_key == ""
    assert config.adversaries[0].llm_config.api_key == ""
    assert config.max_turns == 2


@pytest.mark.asyncio
async def test_mcp_trigger_masc_audit_success(base_masc_config, monkeypatch):
    async def fake_workflow(task_description, config, thread_id):
        yield {
            "synthesize_sequential": {
                "current_artifact": Artifact(
                    version="2.0_sequential",
                    content="Hardened MCP output",
                    history=["Initial proposal", "[Turn 1] Fixed issue"],
                ),
                "current_turn": 2,
            }
        }

    monkeypatch.setattr("src.entrypoints.mcp_server.execute_masc_workflow", fake_workflow)

    result = await trigger_masc_audit(
        task_description="Audit this service",
        config_json=base_masc_config.model_dump_json(),
    )
    assert "MASC Dialectical Audit Complete" in result
    assert "Hardened MCP output" in result
    assert "[Turn 1] Fixed issue" in result


@pytest.mark.asyncio
async def test_model_fetcher_returns_fallbacks_when_no_key():
    for provider in ("OpenAI", "Anthropic", "Google"):
        models = await fetch_available_models(provider, api_key="")
        assert models == FALLBACK_MODELS[provider]