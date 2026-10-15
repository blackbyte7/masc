from unittest.mock import MagicMock

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

from src.config.settings import settings
from src.core.engine import MASCNodes, should_continue_loop, should_route_to_synthesis
from src.core.state import (
    Artifact,
    Critique,
    CritiqueItem,
    CritiqueLLMOutput,
    CritiquesCollection,
    MASCState,
)


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
    nodes._get_llm(base_masc_config.proposer)
    assert captured_kwargs["api_key"] is None


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
async def test_synthesize_sequential_batches_per_adversary(base_masc_config, monkeypatch):
    nodes = MASCNodes(settings)
    fake_llm = FakeListChatModel(
        responses=[
            "Draft after refuting DA.",
            "Draft after fixing all CodeAuditor issues in a single batched pass.",
        ]
    )
    monkeypatch.setattr(nodes, "_get_llm", lambda _cfg: fake_llm)

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
    assert artifact.content == "Draft after fixing all CodeAuditor issues in a single batched pass."
    assert len(artifact.history) == 4


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