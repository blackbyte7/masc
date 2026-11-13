"""Regression tests for the security findings in the production-readiness audit."""

import json
from typing import Any

import httpx
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import PrivateAttr

from src.config.settings import MASCSettings, settings, validate_base_url
from src.core import engine
from src.core.credentials import VAULT_REF_PREFIX, resolve_api_key, seal_config_credentials, vault
from src.core.engine import MASCNodes, execute_masc_workflow, resume_masc_workflow
from src.core.state import CritiqueItem, CritiqueLLMOutput
from src.entrypoints.api_server import reset_rate_limit_buckets
from src.utils import model_fetcher

SECRET = "sk-user-supplied-secret-0123456789"


@pytest.fixture(autouse=True)
def _reset_buckets():
    reset_rate_limit_buckets()
    yield
    reset_rate_limit_buckets()


class _StructuredFake(FakeListChatModel):
    _structured: Any = PrivateAttr(default=None)

    def __init__(self, responses: list[str], structured: Any = None, **kwargs: Any) -> None:
        super().__init__(responses=responses, **kwargs)
        self._structured = structured

    def with_structured_output(self, _schema: Any, **_kwargs: Any) -> Runnable:
        result = self._structured
        return RunnableLambda(lambda _x: result)


def _checkpoint_dump() -> str:
    saver = engine._memory_saver
    return repr(dict(saver.storage)) + repr(dict(saver.writes)) + repr(dict(saver.blobs))


# --- S1: API keys are never persisted to checkpoints -----------------------------------------


@pytest.mark.asyncio
async def test_api_keys_are_not_persisted_in_checkpoints_but_reach_the_llm(base_masc_config, monkeypatch):
    for persona in (base_masc_config.proposer, base_masc_config.synthesizer, *base_masc_config.adversaries):
        persona.llm_config.api_key = SECRET
    base_masc_config.require_human_approval = True

    seen_keys: list[str] = []
    critique = CritiqueLLMOutput(
        payload=[CritiqueItem(severity="HIGH", description="Issue", recommendation="Refute this.")]
    )

    def fake_get_llm(_self, cfg):
        seen_keys.append(cfg.llm_config.api_key)
        return _StructuredFake(responses=["Synthesized"], structured=critique)

    monkeypatch.setattr(MASCNodes, "_get_llm", fake_get_llm)

    thread_id = "security-thread-keys"
    events = [ev async for ev in execute_masc_workflow("Audit", base_masc_config, thread_id, initial_artifact="x")]
    assert "__interrupt__" in events[-1]

    dump = _checkpoint_dump()
    assert SECRET not in dump
    assert VAULT_REF_PREFIX in dump
    # The caller's config object is not mutated
    assert base_masc_config.proposer.llm_config.api_key == SECRET

    # Resume (same process) still has access to the user's key via the vault
    resumed = [ev async for ev in resume_masc_workflow(thread_id=thread_id, approve=True)]
    assert any("synthesize_sequential" in ev for ev in resumed)
    assert seen_keys and all(k == SECRET for k in seen_keys)
    assert SECRET not in _checkpoint_dump()


def test_externally_supplied_vault_reference_is_ignored(base_masc_config):
    victim_ref = vault.put("victim-thread", SECRET)
    base_masc_config.proposer.llm_config.api_key = victim_ref

    sealed = seal_config_credentials(base_masc_config, "victim-thread")
    assert sealed.proposer.llm_config.api_key == ""
    assert resolve_api_key(victim_ref, None) is None


def test_get_llm_never_forwards_vault_reference(base_masc_config, monkeypatch):
    captured: dict[str, Any] = {}

    def fake_init_chat_model(**kwargs):
        captured.update(kwargs)

        class _M:
            def with_retry(self, **_k):
                return self

        return _M()

    monkeypatch.setattr("src.core.engine.init_chat_model", fake_init_chat_model)
    base_masc_config.proposer.llm_config.api_key = f"{VAULT_REF_PREFIX}unknown"
    MASCNodes(settings).get_llm(base_masc_config.proposer)
    assert captured["api_key"] is None


@pytest.mark.asyncio
async def test_thread_state_never_exposes_keys_or_vault_refs(base_masc_config, monkeypatch):
    base_masc_config.proposer.llm_config.api_key = SECRET
    base_masc_config.require_human_approval = True
    critique = CritiqueLLMOutput(payload=[CritiqueItem(severity="LOW", description="d", recommendation="r")])
    monkeypatch.setattr(MASCNodes, "_get_llm", lambda _s, _c: _StructuredFake(responses=["x"], structured=critique))

    thread_id = "security-thread-state"
    _ = [ev async for ev in execute_masc_workflow("Audit", base_masc_config, thread_id, initial_artifact="x")]
    state = await engine.get_thread_checkpoint_state(thread_id)
    dumped = json.dumps(state, default=str)
    assert state["awaiting_human_approval"] is True
    assert SECRET not in dumped
    assert VAULT_REF_PREFIX not in dumped


# --- S2: X-Forwarded-For is only trusted behind a configured proxy ----------------------------


def _ok_workflow(*_a, **_k):
    async def gen():
        yield {"synthesize_sequential": {"current_turn": 2}}

    return gen()


def test_rate_limit_ignores_spoofed_forwarded_for_by_default(test_client, base_masc_config, monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_per_minute", 1)
    monkeypatch.setattr("src.entrypoints.api_server.execute_masc_workflow", _ok_workflow)
    payload = {"task_description": "t", "config": base_masc_config.model_dump()}

    assert test_client.post("/v1/masc/execute", json=payload, headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 200
    resp = test_client.post("/v1/masc/execute", json=payload, headers={"X-Forwarded-For": "2.2.2.2"})
    assert resp.status_code == 429


def test_rate_limit_honours_forwarded_for_when_trusted(test_client, base_masc_config, monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_per_minute", 1)
    monkeypatch.setattr(settings, "trust_proxy_headers", True)
    monkeypatch.setattr("src.entrypoints.api_server.execute_masc_workflow", _ok_workflow)
    payload = {"task_description": "t", "config": base_masc_config.model_dump()}

    assert test_client.post("/v1/masc/execute", json=payload, headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 200
    assert test_client.post("/v1/masc/execute", json=payload, headers={"X-Forwarded-For": "2.2.2.2"}).status_code == 200


# --- S3: OpenAI-compatible endpoint enforces the same guardrails ------------------------------


def test_chat_completions_masc_config_enforces_max_adversaries(test_client, base_masc_config):
    base_masc_config.adversaries = base_masc_config.adversaries * (settings.max_adversaries + 1)
    resp = test_client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "masc_config": base_masc_config.model_dump()},
    )
    assert resp.status_code == 400
    assert "Exceeded max adversaries" in resp.json()["detail"]


def test_chat_completions_masc_config_rejects_unknown_persona(test_client, base_masc_config):
    base_masc_config.adversaries[0].persona_name = "NoSuchPersona"
    resp = test_client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "masc_config": base_masc_config.model_dump()},
    )
    assert resp.status_code == 400
    assert "Unknown adversary persona" in resp.json()["detail"]


def test_chat_completions_rejects_oversized_messages(test_client, monkeypatch):
    monkeypatch.setattr(settings, "max_task_length", 10)
    resp = test_client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "x" * 11}]},
    )
    assert resp.status_code == 413


def test_invalid_thread_id_rejected(test_client, base_masc_config):
    payload = {"task_description": "t", "config": base_masc_config.model_dump(), "thread_id": "../../etc"}
    assert test_client.post("/v1/masc/execute", json=payload).status_code == 422


# --- S4: Google API key is sent as a header, never in the URL ---------------------------------


@pytest.mark.asyncio
async def test_google_model_fetch_sends_key_in_header(monkeypatch):
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["header"] = request.headers.get("x-goog-api-key")
        return httpx.Response(200, json={"models": [{"name": "models/gemini-test"}]})

    real_client = httpx.AsyncClient

    class _MockClient(real_client):  # type: ignore[misc,valid-type]
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(model_fetcher.httpx, "AsyncClient", _MockClient)
    models = await model_fetcher.fetch_available_models("Google", api_key="AIza-secret")
    assert models == ["gemini-test"]
    assert seen["header"] == "AIza-secret"
    assert "AIza-secret" not in seen["url"]


# --- S5: SSRF validation covers alternate IP encodings ----------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/latest",
        "http://[fe80::1]:11434",
        "http://[::ffff:169.254.169.254]/",
        "http://2852039166/",
        "http://0xA9FEA9FE/",
        "http://0.0.0.0:11434",
        "http://metadata.google.internal/",
        "http://user:pass@example.com/v1",
        "ftp://example.com",
    ],
)
def test_validate_base_url_blocks_dangerous_targets_even_when_custom_allowed(url):
    permissive = MASCSettings(allow_custom_base_urls=True)
    with pytest.raises(ValueError):
        validate_base_url(url, permissive)


def test_validate_base_url_allows_loopback_and_allowlist():
    permissive = MASCSettings(allow_custom_base_urls=True)
    assert validate_base_url("http://127.0.0.1:11434/", permissive) == "http://127.0.0.1:11434"
    assert validate_base_url("https://llm.internal.example.com/v1", permissive)
    strict = MASCSettings(allow_custom_base_urls=False)
    assert validate_base_url("http://localhost:11434", strict) == "http://localhost:11434"
    with pytest.raises(ValueError, match="Disallowed base_url"):
        validate_base_url("https://example.com", strict)


# --- Settings -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("postgresql+psycopg://u:p@h:5432/db", "postgresql://u:p@h:5432/db"),
        ("postgresql+asyncpg://u:p@h/db", "postgresql://u:p@h/db"),
        ("postgres://u:p@h/db", "postgres://u:p@h/db"),
        ("memory://", "memory://"),
    ],
)
def test_database_url_normalization(raw, expected):
    assert MASCSettings(database_url=raw).database_url == expected


def test_uses_postgres_flag():
    assert MASCSettings(database_url="postgresql+psycopg://u@h/db").uses_postgres
    assert not MASCSettings(database_url="memory://").uses_postgres


def test_api_resume_on_unpaused_thread_returns_409(test_client, monkeypatch):
    async def fake_state(thread_id):
        return {"thread_id": thread_id, "awaiting_human_approval": False, "next": [], "values": {"current_turn": 2}}

    monkeypatch.setattr("src.entrypoints.api_server.get_thread_checkpoint_state", fake_state)
    resp = test_client.post("/v1/masc/threads/finished-thread/resume", json={"approve": True})
    assert resp.status_code == 409
    assert json.loads(resp.text)["detail"].endswith("not paused awaiting human approval.")
