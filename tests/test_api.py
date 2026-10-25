import json

import pytest

from src.config.settings import settings
from src.core.state import Artifact, TokenUsage
from src.entrypoints.api_server import reset_rate_limit_buckets


@pytest.fixture(autouse=True)
def clear_rate_limit_buckets():
    reset_rate_limit_buckets()
    yield
    reset_rate_limit_buckets()


def test_health_endpoint(test_client):
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_execute_masc_endpoint_validation_error(test_client):
    # Missing required config field
    response = test_client.post("/v1/masc/execute", json={"task_description": "Do something"})
    assert response.status_code == 422


def test_execute_masc_empty_task_description(test_client, base_masc_config):
    payload = {
        "task_description": "",
        "config": base_masc_config.model_dump(),
    }
    response = test_client.post("/v1/masc/execute", json=payload)
    assert response.status_code == 422


def test_execute_masc_max_adversaries_limit(test_client, base_masc_config):
    # Duplicate adversaries to exceed limit (default max is 5)
    for _ in range(10):
        base_masc_config.adversaries.append(base_masc_config.adversaries[0])

    payload = {
        "task_description": "Test high adversaries",
        "config": base_masc_config.model_dump(),
    }

    response = test_client.post("/v1/masc/execute", json=payload)
    assert response.status_code == 400
    assert "Exceeded max adversaries" in response.json()["detail"]


def test_execute_masc_unknown_persona(test_client, base_masc_config):
    base_masc_config.adversaries[0].persona_name = "NonExistentPersona"
    payload = {
        "task_description": "Test unknown persona",
        "config": base_masc_config.model_dump(),
    }

    response = test_client.post("/v1/masc/execute", json=payload)
    assert response.status_code == 400
    assert "Unknown adversary persona" in response.json()["detail"]


def test_execute_masc_accepts_inline_custom_persona(test_client, base_masc_config, monkeypatch):
    async def fake_workflow(_task_description, _config, _thread_id, **_kwargs):
        yield {
            "synthesize_sequential": {
                "current_artifact": Artifact(
                    version="2.0_sequential",
                    content="Audited by inline persona",
                    history=["[Turn 1] Audited by EphemeralFinOpsCritic"],
                ),
                "current_turn": 2,
            }
        }

    monkeypatch.setattr("src.entrypoints.api_server.execute_masc_workflow", fake_workflow)

    base_masc_config.adversaries[0].persona_name = "EphemeralFinOpsCritic"
    base_masc_config.adversaries[0].critique_type = "CONSTRUCTIVE"
    base_masc_config.adversaries[0].system_prompt = "Audit cloud architecture strictly for egress cost risks."

    payload = {
        "task_description": "Design a multi-region data replication pipeline",
        "config": base_masc_config.model_dump(),
    }

    response = test_client.post("/v1/masc/execute", json=payload)
    assert response.status_code == 200
    assert "Audited by inline persona" in response.text


def test_execute_masc_accepts_empty_adversaries_when_auto_router_enabled(
    test_client, base_masc_config, monkeypatch
):
    async def fake_workflow(_task_description, _config, _thread_id, **_kwargs):
        yield {
            "synthesize_sequential": {
                "current_artifact": Artifact(
                    version="2.0_sequential",
                    content="Auto-routed output",
                    history=["[Turn 1] Auto-routed"],
                ),
                "current_turn": 2,
            }
        }

    monkeypatch.setattr("src.entrypoints.api_server.execute_masc_workflow", fake_workflow)

    base_masc_config.adversaries = []
    base_masc_config.auto_select_adversaries = True

    payload = {
        "task_description": "Audit this service with auto-selected personas",
        "config": base_masc_config.model_dump(),
    }
    response = test_client.post("/v1/masc/execute", json=payload)
    assert response.status_code == 200
    assert "Auto-routed output" in response.text


def test_execute_masc_ssrf_base_url_rejected(test_client, base_masc_config):
    base_masc_config.proposer.llm_config.base_url = "http://169.254.169.254/latest/meta-data"
    payload = {
        "task_description": "Test SSRF protection",
        "config": base_masc_config.model_dump(),
    }

    response = test_client.post("/v1/masc/execute", json=payload)
    assert response.status_code == 400
    assert "Disallowed base_url" in response.json()["detail"]


def test_execute_masc_custom_base_url_without_key_rejected(test_client, base_masc_config, monkeypatch):
    monkeypatch.setattr(settings, "allow_custom_base_urls", True)
    base_masc_config.proposer.llm_config.base_url = "https://example.com/v1"
    base_masc_config.proposer.llm_config.api_key = ""
    payload = {
        "task_description": "Test env key exfiltration guard",
        "config": base_masc_config.model_dump(),
    }

    response = test_client.post("/v1/masc/execute", json=payload)
    assert response.status_code == 400
    assert "Explicit API key is required" in response.json()["detail"]


def test_execute_masc_auth_guard(test_client, base_masc_config, monkeypatch):
    monkeypatch.setattr(settings, "masc_api_key", "super-secret-token")
    payload = {
        "task_description": "Test auth",
        "config": base_masc_config.model_dump(),
    }

    # Missing Authorization header -> 401
    resp_unauth = test_client.post("/v1/masc/execute", json=payload)
    assert resp_unauth.status_code == 401

    # Invalid token -> 403
    resp_forbidden = test_client.post(
        "/v1/masc/execute",
        json=payload,
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert resp_forbidden.status_code == 403


def test_execute_masc_sse_stream_and_redaction(test_client, base_masc_config, monkeypatch):
    async def fake_workflow(_task_description, config, _thread_id, **_kwargs):
        yield {
            "propose": {
                "current_artifact": Artifact(
                    version="1.0_proposal",
                    content="Initial proposal text",
                    history=["Initial proposal"],
                ),
                "config": config,
                "current_turn": 1,
            }
        }
        yield {
            "synthesize_sequential": {
                "current_artifact": Artifact(
                    version="2.0_sequential",
                    content="Hardened final text",
                    history=["Initial proposal", "[Turn 1] Hardened against: DA"],
                ),
                "current_turn": 2,
            }
        }

    monkeypatch.setattr("src.entrypoints.api_server.execute_masc_workflow", fake_workflow)

    payload = {
        "task_description": "Test streaming endpoint",
        "config": base_masc_config.model_dump(),
    }

    response = test_client.post("/v1/masc/execute", json=payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    events = [
        line.removeprefix("data: ")
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert events[-1] == "[DONE]"

    first_event = json.loads(events[0])
    assert first_event["node_executed"] == "propose"
    assert first_event["current_turn"] == 1
    # Verify API keys are stripped from streamed config state
    assert "api_key" not in first_event["state_data"]["config"]["proposer"]["llm_config"]

    second_event = json.loads(events[1])
    assert second_event["node_executed"] == "synthesize_sequential"
    # Verify post-synthesis turn is normalized back to the active turn (1)
    assert second_event["current_turn"] == 1


def test_get_thread_state_not_found(test_client):
    response = test_client.get("/v1/masc/threads/non-existent-thread-id")
    assert response.status_code == 404


def test_openai_models_and_chat_completions_endpoints(test_client, monkeypatch):
    models_resp = test_client.get("/v1/models")
    assert models_resp.status_code == 200
    model_ids = [m["id"] for m in models_resp.json()["data"]]
    assert "masc-sequential-default" in model_ids
    assert "masc-auto" in model_ids

    async def fake_workflow(_task_description, _config, _thread_id, **_kwargs):
        yield {
            "synthesize_sequential": {
                "current_artifact": Artifact(
                    version="2.0_sequential",
                    content="OpenAI compatible hardened response",
                    history=["Initial", "Hardened"],
                ),
                "current_turn": 2,
                "token_usage": TokenUsage(input_tokens=50, output_tokens=70, total_tokens=120, llm_calls=3),
            }
        }

    monkeypatch.setattr("src.entrypoints.api_server.execute_masc_workflow", fake_workflow)

    chat_resp = test_client.post(
        "/v1/chat/completions",
        json={
            "model": "masc-sequential-default",
            "messages": [{"role": "user", "content": "Explain zero-trust architecture"}],
            "stream": False,
        },
    )
    assert chat_resp.status_code == 200
    body = chat_resp.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "OpenAI compatible hardened response"
    assert body["usage"]["total_tokens"] == 120


def test_rate_limiting_enforced(test_client, base_masc_config, monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_per_minute", 2)

    async def fake_workflow(_task_description, _config, _thread_id, **_kwargs):
        yield {
            "synthesize_sequential": {
                "current_artifact": Artifact(version="2.0", content="ok", history=[]),
                "current_turn": 2,
            }
        }

    monkeypatch.setattr("src.entrypoints.api_server.execute_masc_workflow", fake_workflow)
    payload = {"task_description": "Test rate limit", "config": base_masc_config.model_dump()}

    assert test_client.post("/v1/masc/execute", json=payload).status_code == 200
    assert test_client.post("/v1/masc/execute", json=payload).status_code == 200
    resp_limited = test_client.post("/v1/masc/execute", json=payload)
    assert resp_limited.status_code == 429
