import json

from src.core.personas import load_personas


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