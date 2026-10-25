import json
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import PrivateAttr

from src.config.settings import settings
from src.core.engine import (
    MASCNodes,
    execute_masc_workflow,
    resume_masc_workflow,
    should_continue_loop,
    should_propose_or_audit,
    should_route_after_propose,
    should_route_to_synthesis,
)
from src.core.personas import run_deterministic_verification
from src.core.state import (
    Artifact,
    Critique,
    CritiqueItem,
    CritiqueLLMOutput,
    CritiquesCollection,
    MASCState,
    PersonaRouterOutput,
    TokenUsage,
)


class _StructuredFakeChatModel(FakeListChatModel):
    """Pydantic-compatible FakeListChatModel subclass that supports .with_structured_output()."""

    _structured_result: Any = PrivateAttr(default=None)

    def __init__(self, responses: list[str], structured_result: Any = None, **kwargs: Any) -> None:
        super().__init__(responses=responses, **kwargs)
        self._structured_result = structured_result

    def with_structured_output(self, _schema: Any, **_kwargs: Any) -> Runnable:
        result = self._structured_result
        return RunnableLambda(lambda _x: result)


@pytest.mark.asyncio
async def test_should_propose_or_audit_routing(base_masc_config):
    state_empty: MASCState = {
        "task_description": "Test",
        "config": base_masc_config,
        "current_artifact": Artifact(version="0.0", content=""),
        "current_turn": 1,
    }
    assert should_propose_or_audit(state_empty) == "propose"
    assert should_route_after_propose(state_empty) == "adversarial_analysis"

    state_with_artifact: MASCState = {
        "task_description": "Audit existing code",
        "config": base_masc_config,
        "current_artifact": Artifact(version="1.0_initial", content="def foo(): pass"),
        "current_turn": 1,
    }
    assert should_propose_or_audit(state_with_artifact) == "adversarial_analysis"

    # When optional auto_select_adversaries is True, route via route_personas first
    base_masc_config.auto_select_adversaries = True
    assert should_propose_or_audit(state_with_artifact) == "route_personas"
    assert should_route_after_propose(state_empty) == "route_personas"


@pytest.mark.asyncio
async def test_optional_route_personas_node(base_masc_config, monkeypatch):
    nodes = MASCNodes(settings)
    router_output = PersonaRouterOutput(
        selected_personas=["SecurityHawk", "CodeAuditor"],
        reasoning="Task involves code and security.",
    )
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = RunnableLambda(lambda _x: router_output)
    monkeypatch.setattr(nodes, "_get_llm", lambda _cfg: mock_llm)

    base_masc_config.adversaries = []
    base_masc_config.auto_select_adversaries = True

    state: MASCState = {
        "task_description": "Audit this authentication microservice.",
        "config": base_masc_config,
        "current_artifact": Artifact(version="1.0", content="code"),
        "current_turn": 1,
    }

    result = await nodes.route_personas(state)
    selected = [adv.persona_name for adv in result["config"].adversaries]
    assert selected == ["SecurityHawk", "CodeAuditor"]
    assert result["token_usage"].llm_calls == 1


def test_deterministic_verification_helper():
    bad_code_markdown = (
        "Here is the implementation:\n"
        "```python\n"
        "import subprocess\n"
        "API_KEY = 'sk-1234567890abcdefghijklmnop'\n"
        "eval(user_input)\n"
        "subprocess.run('ls', shell=True)\n"
        "```\n"
    )
    findings = run_deterministic_verification("CodeAuditor", bad_code_markdown)
    assert any("STATIC SECRET SCAN" in f for f in findings)
    assert any("eval()" in f for f in findings)
    assert any("shell=True" in f for f in findings)


@pytest.mark.asyncio
async def test_routing_to_synthesis(base_masc_config):
    # Test Sequential Refinement
    state_sequential: MASCState = {
        "task_description": "Test",
        "config": base_masc_config,
        "current_artifact": Artifact(version="1.0", content="test"),
        "current_turn": 1,
        "critiques_collection": None,
    }
    assert should_route_to_synthesis(state_sequential) == "triage_critiques"

    # Test Architect Protocol
    base_masc_config.synthesis_protocol = "Architect"
    state_architect: MASCState = {
        "task_description": "Test",
        "config": base_masc_config,
        "current_artifact": Artifact(version="1.0", content="test"),
        "current_turn": 1,
        "critiques_collection": None,
    }
    assert should_route_to_synthesis(state_architect) == "synthesize_architect"


@pytest.mark.asyncio
async def test_loop_continuation(base_masc_config):
    state: MASCState = {
        "task_description": "Test",
        "config": base_masc_config,
        "current_artifact": Artifact(version="1.0", content="test"),
        "current_turn": 1,
        "critiques_collection": None,
    }

    # If current turn is <= max_turns, it should loop
    assert should_continue_loop(state) == "adversarial_analysis"

    # If turn exceeds max_turns, it should end
    state["current_turn"] = 2
    assert should_continue_loop(state) == "__end__"


@pytest.mark.asyncio
async def test_early_convergence_when_no_actionable_critiques(base_masc_config):
    base_masc_config.max_turns = 5
    state: MASCState = {
        "task_description": "Test",
        "config": base_masc_config,
        "current_artifact": Artifact(version="2.0", content="hardened"),
        "current_turn": 2,
        "critiques_collection": CritiquesCollection(critiques=[]),
    }
    # Even though current_turn (2) <= max_turns (5), empty critiques trigger early convergence
    assert should_continue_loop(state) == "__end__"


@pytest.mark.asyncio
async def test_severity_threshold_convergence_and_token_budget_breaker(base_masc_config):
    base_masc_config.max_turns = 5
    base_masc_config.stop_on_max_severity = "HIGH"

    low_only_critiques = CritiquesCollection(
        critiques=[
            Critique(
                source_adversary="CodeAuditor",
                critique_type="CONSTRUCTIVE",
                payload=[CritiqueItem(severity="LOW", description="Nitpick", recommendation="Optional")],
            )
        ]
    )
    state: MASCState = {
        "task_description": "Test",
        "config": base_masc_config,
        "current_artifact": Artifact(version="2.0", content="hardened"),
        "current_turn": 2,
        "critiques_collection": low_only_critiques,
    }
    # Stops early because only LOW severity remains while threshold is HIGH
    assert should_continue_loop(state) == "__end__"

    # Now test token budget circuit breaker when HIGH severity exists
    high_critiques = CritiquesCollection(
        critiques=[
            Critique(
                source_adversary="CodeAuditor",
                critique_type="CONSTRUCTIVE",
                payload=[CritiqueItem(severity="HIGH", description="Major bug", recommendation="Fix")],
            )
        ]
    )
    base_masc_config.max_total_tokens = 1000
    state["critiques_collection"] = high_critiques
    state["token_usage"] = TokenUsage(input_tokens=600, output_tokens=500, total_tokens=1100, llm_calls=3)
    assert should_continue_loop(state) == "__end__"


def test_get_llm_treats_env_placeholder_as_none(base_masc_config, monkeypatch):
    nodes = MASCNodes(settings)
    captured_kwargs = {}

    def fake_init_chat_model(**kwargs):
        captured_kwargs.update(kwargs)
        mock_model = MagicMock()
        mock_model.with_retry.return_value = mock_model
        return mock_model

    monkeypatch.setattr("src.core.engine.init_chat_model", fake_init_chat_model)

    base_masc_config.proposer.llm_config.api_key = "env"
    nodes.get_llm(base_masc_config.proposer)
    assert captured_kwargs["api_key"] is None


def test_get_llm_blocks_env_key_exfiltration_with_custom_base_url(base_masc_config):
    nodes = MASCNodes(settings)
    base_masc_config.proposer.llm_config.api_key = ""
    base_masc_config.proposer.llm_config.base_url = "http://localhost:11434"

    with pytest.raises(ValueError, match="Explicit API key is required"):
        nodes.get_llm(base_masc_config.proposer)


@pytest.mark.asyncio
async def test_deterministic_triage_sorting(base_masc_config):
    nodes = MASCNodes(settings)
    unordered = CritiquesCollection(
        critiques=[
            Critique(
                source_adversary="CodeAuditor",
                critique_type="CONSTRUCTIVE",
                payload=[
                    CritiqueItem(severity="LOW", description="Minor nit", recommendation="Fix style"),
                    CritiqueItem(severity="CRITICAL", description="SQLi", recommendation="Parameterize"),
                ],
            ),
            Critique(
                source_adversary="DA",
                critique_type="ANTAGONISTIC",
                payload=[
                    CritiqueItem(
                        severity="HIGH",
                        description="Core thesis is flawed",
                        recommendation="Refute this ideological challenge.",
                    )
                ],
            ),
        ]
    )

    state: MASCState = {
        "task_description": "Test",
        "config": base_masc_config,
        "current_artifact": Artifact(version="1.0", content="test"),
        "current_turn": 1,
        "critiques_collection": unordered,
    }

    result = await nodes.triage_critiques(state)
    sorted_critiques = result["critiques_collection"].critiques

    assert sorted_critiques[0].source_adversary == "DA"
    assert sorted_critiques[1].payload[0].severity == "CRITICAL"
    assert sorted_critiques[1].payload[1].severity == "LOW"
    assert result["current_turn"] == 1


@pytest.mark.asyncio
async def test_propose_node(base_masc_config, monkeypatch):
    nodes = MASCNodes(settings)
    fake_llm = FakeListChatModel(responses=["Initial drafted proposal content."])
    monkeypatch.setattr(nodes, "_get_llm", lambda _cfg: fake_llm)

    state: MASCState = {
        "task_description": "Design a zero-trust microservice architecture.",
        "config": base_masc_config,
        "current_artifact": Artifact(version="0.0", content=""),
        "current_turn": 1,
        "critiques_collection": None,
    }

    result = await nodes.propose(state)
    assert result["current_turn"] == 1
    assert result["current_artifact"].version == "1.0_proposal"
    assert result["current_artifact"].content == "Initial drafted proposal content."
    assert len(result["current_artifact"].history) == 1
    assert result["token_usage"].llm_calls == 1


@pytest.mark.asyncio
async def test_adversarial_analysis_node(base_masc_config, monkeypatch):
    nodes = MASCNodes(settings)
    expected_output = CritiqueLLMOutput(
        payload=[
            CritiqueItem(
                severity="HIGH",
                description="Assumes internal network is trusted.",
                recommendation="Refute this ideological challenge.",
            )
        ]
    )

    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = RunnableLambda(lambda _x: expected_output)
    monkeypatch.setattr(nodes, "_get_llm", lambda _cfg: mock_llm)

    state: MASCState = {
        "task_description": "Design a zero-trust microservice architecture.",
        "config": base_masc_config,
        "current_artifact": Artifact(version="1.0_proposal", content="Initial proposal"),
        "current_turn": 1,
        "critiques_collection": None,
    }

    result = await nodes.adversarial_analysis(state)
    collection = result["critiques_collection"]
    assert result["current_turn"] == 1
    assert len(collection.critiques) == 1
    assert collection.critiques[0].source_adversary == "DA"
    assert collection.critiques[0].critique_type == "ANTAGONISTIC"
    assert collection.critiques[0].payload[0].severity == "HIGH"
    assert result["token_usage"].llm_calls == 1


@pytest.mark.asyncio
async def test_adversarial_analysis_handles_curly_braces_in_system_prompt(base_masc_config, monkeypatch):
    nodes = MASCNodes(settings)
    expected_output = CritiqueLLMOutput(
        payload=[
            CritiqueItem(
                severity="MEDIUM",
                description="Missing JSON validation.",
                recommendation="Validate against schema.",
            )
        ]
    )

    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = RunnableLambda(lambda _x: expected_output)
    monkeypatch.setattr(nodes, "_get_llm", lambda _cfg: mock_llm)

    # Ensure unescaped JSON braces in system_prompt do not trigger LangChain KeyError
    persona_with_braces = {
        "critique_type": "CONSTRUCTIVE",
        "system_prompt": 'Audit this JSON structure: {"user_id": 123, "role": "admin"}',
    }
    critique, _usage = await nodes.run_single_adversary(
        adv_config=base_masc_config.adversaries[0],
        name="CustomJSONAuditor",
        persona=persona_with_braces,
        task_description="Build a user service",
        content="Draft service",
    )
    assert critique.source_adversary == "CustomJSONAuditor"
    assert len(critique.payload) == 1


@pytest.mark.asyncio
async def test_adversarial_analysis_raises_when_all_fail(base_masc_config, monkeypatch):
    nodes = MASCNodes(settings)

    def failing_llm(_cfg):
        raise RuntimeError("Provider connection failed")

    monkeypatch.setattr(nodes, "_get_llm", failing_llm)

    state: MASCState = {
        "task_description": "Test failure",
        "config": base_masc_config,
        "current_artifact": Artifact(version="1.0_proposal", content="Initial proposal"),
        "current_turn": 1,
        "critiques_collection": None,
    }

    with pytest.raises(RuntimeError, match="All configured adversaries failed"):
        await nodes.adversarial_analysis(state)


@pytest.mark.asyncio
async def test_synthesize_sequential_batches_per_adversary_and_enforces_response_schema(
    base_masc_config, monkeypatch
):
    nodes = MASCNodes(settings)
    fake_llm = _StructuredFakeChatModel(
        responses=[
            "Draft after refuting DA.",
            "Draft after fixing all CodeAuditor issues in a single batched pass.",
        ],
        structured_result={"service": "hardened_db_query", "parameterized": True},
    )
    monkeypatch.setattr(nodes, "_get_llm", lambda _cfg: fake_llm)

    base_masc_config.response_schema = {
        "title": "ServiceSpec",
        "type": "object",
        "properties": {"service": {"type": "string"}, "parameterized": {"type": "boolean"}},
    }

    critiques = CritiquesCollection(
        critiques=[
            Critique(
                source_adversary="DA",
                critique_type="ANTAGONISTIC",
                payload=[
                    CritiqueItem(
                        severity="HIGH",
                        description="Existential critique",
                        recommendation="Refute this ideological challenge.",
                    )
                ],
            ),
            Critique(
                source_adversary="CodeAuditor",
                critique_type="CONSTRUCTIVE",
                payload=[
                    CritiqueItem(severity="CRITICAL", description="SQL Injection", recommendation="Use bind params"),
                    CritiqueItem(severity="MEDIUM", description="Missing timeout", recommendation="Set timeout=10"),
                ],
            ),
        ]
    )

    state: MASCState = {
        "task_description": "Write a database query service.",
        "config": base_masc_config,
        "current_artifact": Artifact(version="1.0_proposal", content="v1 content", history=["Initial"]),
        "current_turn": 1,
        "critiques_collection": critiques,
    }

    result = await nodes.synthesize_sequential(state)
    artifact = result["current_artifact"]

    assert result["current_turn"] == 2
    assert artifact.version == "2.0_sequential"
    parsed_json = json.loads(artifact.content)
    assert parsed_json == {"service": "hardened_db_query", "parameterized": True}
    assert result["token_usage"].llm_calls == 3


@pytest.mark.asyncio
async def test_synthesize_sequential_raises_when_all_fail(base_masc_config, monkeypatch):
    nodes = MASCNodes(settings)
    failing_llm = RunnableLambda(lambda _x: (_ for _ in ()).throw(RuntimeError("LLM quota exceeded")))
    monkeypatch.setattr(nodes, "_get_llm", lambda _cfg: failing_llm)

    critiques = CritiquesCollection(
        critiques=[
            Critique(
                source_adversary="DA",
                critique_type="ANTAGONISTIC",
                payload=[
                    CritiqueItem(
                        severity="HIGH",
                        description="Existential critique",
                        recommendation="Refute this ideological challenge.",
                    )
                ],
            )
        ]
    )

    state: MASCState = {
        "task_description": "Test total synthesis failure.",
        "config": base_masc_config,
        "current_artifact": Artifact(version="1.0_proposal", content="v1 content", history=["Initial"]),
        "current_turn": 1,
        "critiques_collection": critiques,
    }

    with pytest.raises(RuntimeError, match="All sequential synthesis steps failed"):
        await nodes.synthesize_sequential(state)


@pytest.mark.asyncio
async def test_synthesize_architect_node(base_masc_config, monkeypatch):
    nodes = MASCNodes(settings)
    fake_llm = FakeListChatModel(responses=["Holistically rewritten architect proposal."])
    monkeypatch.setattr(nodes, "_get_llm", lambda _cfg: fake_llm)

    critiques = CritiquesCollection(
        critiques=[
            Critique(
                source_adversary="UncertaintyQuantifier",
                critique_type="CONSTRUCTIVE",
                payload=[
                    CritiqueItem(
                        severity="HIGH",
                        description="Unverified latency claims",
                        recommendation="Add P99 benchmarks",
                    )
                ],
            )
        ]
    )

    state: MASCState = {
        "task_description": "Design a low-latency trading engine.",
        "config": base_masc_config,
        "current_artifact": Artifact(version="1.0_proposal", content="v1 content", history=["Initial"]),
        "current_turn": 1,
        "critiques_collection": critiques,
    }

    result = await nodes.synthesize_architect(state)
    artifact = result["current_artifact"]

    assert result["current_turn"] == 2
    assert artifact.version == "2.0_architect"
    assert artifact.content == "Holistically rewritten architect proposal."
    assert len(artifact.history) == 3
    assert result["token_usage"].llm_calls == 1


@pytest.mark.asyncio
async def test_human_in_the_loop_interrupt_and_resume_with_override(base_masc_config, monkeypatch):
    base_masc_config.require_human_approval = True
    expected_critique = CritiqueLLMOutput(
        payload=[
            CritiqueItem(
                severity="HIGH",
                description="Needs human review before patching.",
                recommendation="Refute this ideological challenge.",
            )
        ]
    )

    fake_chat = _StructuredFakeChatModel(
        responses=["Synthesized after human approval"],
        structured_result=expected_critique,
    )
    monkeypatch.setattr(MASCNodes, "_get_llm", lambda _self, _cfg: fake_chat)

    thread_id = "test-hitl-thread-override"
    events = []
    async for ev in execute_masc_workflow(
        "Audit this document",
        base_masc_config,
        thread_id,
        initial_artifact="Initial draft",
    ):
        events.append(ev)

    assert "__interrupt__" in events[-1]
    assert events[-1]["__interrupt__"]["awaiting_human_approval"] is True

    edited_critiques = CritiquesCollection(
        critiques=[
            Critique(
                source_adversary="HumanReviewer",
                critique_type="CONSTRUCTIVE",
                payload=[
                    CritiqueItem(
                        severity="CRITICAL",
                        description="Human-curated security issue",
                        recommendation="Add mTLS everywhere",
                    )
                ],
            )
        ]
    )

    resumed_events = []
    async for ev in resume_masc_workflow(
        thread_id=thread_id,
        critiques_override=edited_critiques,
        approve=True,
    ):
        resumed_events.append(ev)

    synth_events = [ev["synthesize_sequential"] for ev in resumed_events if "synthesize_sequential" in ev]
    assert len(synth_events) == 1
    assert any("HumanReviewer" in h for h in synth_events[0]["current_artifact"].history)
