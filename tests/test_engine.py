import pytest
from src.core.state import MASCState, Artifact, CritiquesCollection, Critique
from src.core.engine import should_route_to_synthesis, should_continue_loop


@pytest.mark.asyncio
async def test_routing_to_synthesis(base_masc_config):
    # Test Sequential Refinement
    state_sequential: MASCState = {
        "task_description": "Test",
        "config": base_masc_config,
        "current_artifact": Artifact(version="1.0", content="test"),
        "current_turn": 1,
        "critiques_collection": None
    }
    assert should_route_to_synthesis(state_sequential) == "triage_critiques"

    # Test Architect Protocol
    base_masc_config.synthesis_protocol = "Architect"
    state_architect: MASCState = {
        "task_description": "Test",
        "config": base_masc_config,
        "current_artifact": Artifact(version="1.0", content="test"),
        "current_turn": 1,
        "critiques_collection": None
    }
    assert should_route_to_synthesis(state_architect) == "synthesize_architect"


@pytest.mark.asyncio
async def test_loop_continuation(base_masc_config):
    state: MASCState = {
        "task_description": "Test",
        "config": base_masc_config,
        "current_artifact": Artifact(version="1.0", content="test"),
        "current_turn": 1,  # Max turns is 1
        "critiques_collection": None
    }

    # If current turn is <= max_turns, it should loop
    assert should_continue_loop(state) == "adversarial_analysis"

    # If turn exceeds max_turns, it should end
    state["current_turn"] = 2
    assert should_continue_loop(state) == "__end__"