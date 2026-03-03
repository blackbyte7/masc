import json
import os

from src.core.personas import load_personas


def test_default_personas_loaded():
    personas = load_personas()
    assert "UncertaintyQuantifier" in personas
    assert "DA" in personas
    assert personas["DA"]["critique_type"] == "ANTAGONISTIC"


def test_custom_personas_loading(tmp_path):
    custom_personas = {
        "TestAdversary": {
            "critique_type": "CONSTRUCTIVE",
            "system_prompt": "You are a test adversary."
        }
    }

    custom_file = tmp_path / "custom_personas.json"
    custom_file.write_text(json.dumps(custom_personas))

    # Temporarily override the environment variable to point to the temp file
    os.environ["MASC_PERSONAS_FILE"] = str(custom_file)

    try:
        personas = load_personas()
        assert "TestAdversary" in personas
        assert personas["TestAdversary"]["critique_type"] == "CONSTRUCTIVE"
        assert "DA" in personas  # Ensure defaults are still there
    finally:
        del os.environ["MASC_PERSONAS_FILE"]
