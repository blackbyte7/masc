"""Regression tests for functional bugs found in the production-readiness audit."""

import inspect
import json

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from src.core import engine
from src.core.engine import _build_langfuse_callback, _TokenUsageCallback
from src.core.personas import run_deterministic_verification
from src.core.state import Artifact, CritiquesCollection, TokenUsage
from src.entrypoints import mcp_server

# --- B3: token usage is not double counted ----------------------------------------------------


@pytest.mark.asyncio
async def test_token_callback_does_not_double_count_openai_style_results():
    cb = _TokenUsageCallback()
    msg = AIMessage(content="x", usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
    result = LLMResult(
        generations=[[ChatGeneration(message=msg)]],
        llm_output={"token_usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
    )
    await cb.on_llm_end(result)
    assert (cb.input_tokens, cb.output_tokens, cb.total_tokens) == (10, 5, 15)


@pytest.mark.asyncio
async def test_token_callback_falls_back_to_llm_output():
    cb = _TokenUsageCallback()
    result = LLMResult(
        generations=[[ChatGeneration(message=AIMessage(content="x"))]],
        llm_output={"usage": {"input_tokens": 7, "output_tokens": 3}},
    )
    await cb.on_llm_end(result)
    assert (cb.input_tokens, cb.output_tokens, cb.total_tokens) == (7, 3, 10)


# --- B2: Langfuse -------------------------------------------------------------------------------


def test_langfuse_disabled_without_credentials():
    assert _build_langfuse_callback() == (None, None)


# --- B1: Gradio HITL handlers must be detectable as streaming handlers ------------------------


def test_gradio_hitl_handlers_are_async_generator_functions():
    from src.entrypoints import gradio_ui

    assert inspect.isasyncgenfunction(gradio_ui.approve_and_resume)
    assert inspect.isasyncgenfunction(gradio_ui.reject_and_abort)

    fns = {
        getattr(dep.fn, "__name__", None)
        for dep in gradio_ui.app.fns.values()  # type: ignore[attr-defined]
    }
    assert {"approve_and_resume", "reject_and_abort"} <= fns
    assert "<lambda>" not in fns


# --- B6 / B13: MCP server ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mcp_tool_call_does_not_close_shared_pool(base_masc_config, monkeypatch):
    closed = []

    async def fake_close():
        closed.append(True)

    async def fake_workflow(*_args, **_kwargs):
        yield {"synthesize_sequential": {"current_artifact": Artifact(version="2", content="ok"), "current_turn": 2}}

    monkeypatch.setattr(engine, "close_pg_pool", fake_close)
    monkeypatch.setattr(mcp_server, "close_pg_pool", fake_close)
    monkeypatch.setattr(mcp_server, "execute_masc_workflow", fake_workflow)

    result = await mcp_server.trigger_masc_audit("Audit", base_masc_config.model_dump_json())
    assert "Audit Complete" in result
    assert closed == []


@pytest.mark.asyncio
async def test_mcp_resume_reports_second_pause(monkeypatch):
    async def fake_resume(*_args, **_kwargs):
        yield {
            "__interrupt__": {
                "awaiting_human_approval": True,
                "next_nodes": ["synthesize_sequential"],
                "current_turn": 2,
                "critiques_collection": CritiquesCollection(critiques=[]),
            }
        }

    monkeypatch.setattr(mcp_server, "resume_masc_workflow", fake_resume)
    result = await mcp_server.resume_masc_audit(thread_id="t-1", approve=True)
    assert "Paused Again" in result
    assert "turn: 2" in result


@pytest.mark.asyncio
async def test_mcp_validation_errors_are_reported(base_masc_config):
    base_masc_config.adversaries[0].persona_name = "Nope"
    result = await mcp_server.trigger_masc_audit("Audit", base_masc_config.model_dump_json())
    assert result.startswith("MASC Validation Error")
    assert "Unknown adversary persona" in result


# --- B7: in-memory eviction removes blobs as well --------------------------------------------


@pytest.mark.asyncio
async def test_memory_eviction_removes_all_thread_data(base_masc_config, monkeypatch):
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from langchain_core.runnables import RunnableLambda

    from src.core.state import CritiqueLLMOutput

    class _Fake(FakeListChatModel):
        def with_structured_output(self, *_a, **_k):
            return RunnableLambda(lambda _x: CritiqueLLMOutput(payload=[]))

    monkeypatch.setattr(engine.MASCNodes, "_get_llm", lambda _s, _c: _Fake(responses=["x"] * 5))
    monkeypatch.setattr(engine.settings, "max_in_memory_threads", 1)
    engine._memory_thread_order.clear()

    _ = [ev async for ev in engine.execute_masc_workflow("t", base_masc_config, "evict-a", initial_artifact="x")]
    _ = [ev async for ev in engine.execute_masc_workflow("t", base_masc_config, "evict-b", initial_artifact="x")]

    saver = engine._memory_saver
    assert "evict-a" not in saver.storage
    assert not [k for k in saver.blobs if k[0] == "evict-a"]
    assert not [k for k in saver.writes if k[0] == "evict-a"]
    assert "evict-b" in saver.storage


# --- B8: OpenAI streaming emits an error chunk and [DONE] on failure -------------------------


def test_openai_streaming_reports_errors_and_terminates(test_client, monkeypatch):
    async def failing_workflow(*_args, **_kwargs):
        raise RuntimeError("provider exploded")
        yield  # pragma: no cover

    monkeypatch.setattr("src.entrypoints.api_server.execute_masc_workflow", failing_workflow)
    resp = test_client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert resp.status_code == 200
    events = [line.removeprefix("data: ") for line in resp.text.splitlines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    assert any("error" in json.loads(e) for e in events[:-1])
    assert "provider exploded" not in resp.text


def test_openai_streaming_success_includes_usage(test_client, monkeypatch):
    async def ok_workflow(*_args, **_kwargs):
        yield {
            "synthesize_sequential": {
                "current_artifact": Artifact(version="2", content="final answer"),
                "token_usage": TokenUsage(input_tokens=1, output_tokens=2, total_tokens=3, llm_calls=1),
                "current_turn": 2,
            }
        }

    monkeypatch.setattr("src.entrypoints.api_server.execute_masc_workflow", ok_workflow)
    resp = test_client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )
    events = [line.removeprefix("data: ") for line in resp.text.splitlines() if line.startswith("data: ")]
    content = json.loads(events[-2])
    assert content["choices"][0]["delta"]["content"] == "final answer"
    assert content["usage"]["total_tokens"] == 3


# --- B14: CRLF fenced code blocks are verified ------------------------------------------------


def test_deterministic_verification_handles_crlf_code_fences():
    findings = run_deterministic_verification("CodeAuditor", "```python\r\neval(x)\r\n```\r\n")
    assert any("eval()" in f for f in findings)


# --- Convergence logic is shared between the loop router and response_schema enforcement ------


def test_is_final_turn_matches_should_continue_loop(base_masc_config):
    nodes = engine.MASCNodes(engine.settings)
    base_masc_config.max_turns = 3
    empty = CritiquesCollection(critiques=[])
    state = {"config": base_masc_config, "current_turn": 2, "critiques_collection": empty}
    assert engine.should_continue_loop(state) == "__end__"
    assert nodes._is_final_turn(2, base_masc_config, empty, TokenUsage()) is True
