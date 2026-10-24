import json
import pytest

from src.core.personas import load_personas, resolve_adversary_persona
from src.core.state import LLMConfig, PersonaConfig


def test_default_personas_loaded():
    personas = load_personas()
    assert "UncertaintyQuantifier" in personas
    assert "DA" in personas
    assert "SecurityHawk" in personas
    assert personas["DA"]["critique_type"] == "ANTAGONISTIC"
    # Verify SecurityHawk prompt is no longer truncated
    assert "recommendation" in personas["SecurityHawk"]["system_prompt"]
    assert not personas["SecurityHawk"]["system_prompt"].endswith("...")


def test_custom_personas_loading_and_validation(tmp_path, monkeypatch):
    custom_personas = {
        "TestAdversary": {
            "critique_type": "CONSTRUCTIVE",
            "system_prompt": "You are a test adversary.",
        },
        "InvalidCritiqueTypePersona": {
            "critique_type": "UNKNOWN_TYPE",
            "system_prompt": "Should be skipped.",
        },
        "EmptyPromptPersona": {
            "critique_type": "CONSTRUCTIVE",
            "system_prompt": "   ",
        },
    }

    custom_file = tmp_path / "custom_personas.json"
    custom_file.write_text(json.dumps(custom_personas), encoding="utf-8")

    monkeypatch.setenv("MASC_PERSONAS_FILE", str(custom_file))

    personas = load_personas()
    assert "TestAdversary" in personas
    assert personas["TestAdversary"]["critique_type"] == "CONSTRUCTIVE"
    assert "InvalidCritiqueTypePersona" not in personas
    assert "EmptyPromptPersona" not in personas
    assert "DA" in personas


def test_inline_persona_resolution():
    llm_cfg = LLMConfig(provider="OpenAI", model_name="gpt-5.2-pro")
    inline_cfg = PersonaConfig(
        persona_name="InlineK8sAuditor",
        llm_config=llm_cfg,
        critique_type="CONSTRUCTIVE",
        system_prompt="Audit Kubernetes manifests for pod security standards.",
    )
    resolved = resolve_adversary_persona(inline_cfg)
    assert resolved["critique_type"] == "CONSTRUCTIVE"
    assert "Kubernetes" in resolved["system_prompt"]

    with pytest.raises(ValueError, match="both 'critique_type' and a non-empty 'system_prompt'"):
        PersonaConfig(
            persona_name="IncompleteInline",
            llm_config=llm_cfg,
            critique_type="CONSTRUCTIVE",
        )
