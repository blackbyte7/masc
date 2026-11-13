import pytest

from src.core.state import Artifact, MASCConfig
from src.entrypoints.cli import generate_template
from src.entrypoints.mcp_server import resume_masc_audit, trigger_masc_audit
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
    async def fake_workflow(*_args, **_kwargs):
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
async def test_mcp_resume_masc_audit_success(monkeypatch):
    async def fake_resume(*_args, **_kwargs):
        yield {
            "synthesize_sequential": {
                "current_artifact": Artifact(
                    version="2.0_sequential",
                    content="Resumed MCP output",
                    history=["[Turn 1] Resumed after human review"],
                ),
                "current_turn": 2,
            }
        }

    monkeypatch.setattr("src.entrypoints.mcp_server.resume_masc_workflow", fake_resume)

    result = await resume_masc_audit(thread_id="thread-123", approve=True)
    assert "MASC Resumed Audit Complete" in result
    assert "Resumed MCP output" in result


@pytest.mark.asyncio
async def test_model_fetcher_returns_fallbacks_when_no_key():
    for provider in ("OpenAI", "Anthropic", "Google"):
        models = await fetch_available_models(provider, api_key="")
        assert models == FALLBACK_MODELS[provider]


@pytest.mark.asyncio
async def test_model_fetcher_blocks_disallowed_base_url():
    models = await fetch_available_models("Ollama", base_url="http://169.254.169.254/latest")
    assert models == FALLBACK_MODELS["Ollama"]


@pytest.mark.asyncio
async def test_cli_does_not_auto_approve_hitl_without_flag(monkeypatch):
    from src.core.state import CritiquesCollection
    from src.entrypoints import cli

    resumed = []

    async def fake_stream():
        yield {"__interrupt__": {"current_turn": 1, "critiques_collection": CritiquesCollection(critiques=[])}}

    def fake_resume(**kwargs):
        resumed.append(kwargs)
        raise AssertionError("must not resume without --approve-all in a non-interactive session")

    monkeypatch.setattr(cli, "resume_masc_workflow", fake_resume)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False, raising=False)

    content, usage = await cli._consume_cli_stream(fake_stream(), thread_id="t-cli", auto_approve=False)
    assert content == "" and usage is None
    assert resumed == []


@pytest.mark.asyncio
async def test_cli_auto_approves_with_flag(monkeypatch):
    from src.core.state import CritiquesCollection
    from src.entrypoints import cli

    async def fake_stream():
        yield {"__interrupt__": {"current_turn": 1, "critiques_collection": CritiquesCollection(critiques=[])}}

    async def fake_resume(**_kwargs):
        yield {
            "synthesize_sequential": {"current_artifact": Artifact(version="2", content="approved"), "current_turn": 2}
        }

    monkeypatch.setattr(cli, "resume_masc_workflow", fake_resume)
    content, _ = await cli._consume_cli_stream(fake_stream(), thread_id="t-cli", auto_approve=True)
    assert content == "approved"


@pytest.mark.asyncio
async def test_cli_rejects_invalid_config(tmp_path, base_masc_config):
    from src.entrypoints import cli

    base_masc_config.adversaries[0].persona_name = "DoesNotExist"
    cfg_file = tmp_path / "cfg.json"
    cfg_file.write_text(base_masc_config.model_dump_json(), encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        await cli.run_cli("Audit", str(cfg_file))
    assert exc.value.code == 1
