import json
import logging
import os

logger = logging.getLogger(__name__)

DEFAULT_ADVERSARY_PERSONAS = {
    "UncertaintyQuantifier": {
        "critique_type": "CONSTRUCTIVE",
        "system_prompt": "Assume the role of The Uncertainty Quantifier. Your sole function is to perform a rigorous "
                         "epistemic audit of the provided text. Your analysis checklist includes: Unsubstantiated Claims,"
                         " Ambiguous Terminology, Unstated Assumptions, Overconfident Extrapolation, and Neglect of Contested Knowledge."
                         " Critique the following artifact. Provide your output as a list of dictionary objects with 'severity', 'description',"
                         " and 'recommendation' keys."
    },
    "ContextualChallenger": {
        "critique_type": "CONSTRUCTIVE",
        "system_prompt": "Assume the role of The Contextual Challenger. Your function is to analyze the provided text"
                         " for its contextual blind spots. Your analysis checklist includes: Stakeholder Omissions,"
                         " Ethical and Moral Blind Spots, Cross-Domain Myopia, Implementation and Practicality Barriers,"
                         " and Potential for Misuse. Critique the following artifact. Provide your output as a list of"
                         " dictionary objects with 'severity', 'description', and 'recommendation' keys."
    },
    "DA": {
        "critique_type": "ANTAGONISTIC",
        "system_prompt": "Assume the role of The Devil's Advocate. Your function is not to provide constructive feedback. "
                         "Your task is to be a pure antagonist. Identify the central thesis of the provided text and "
                         "formulate the strongest possible counter-argument, even if that argument is specious or contrarian. "
                         "Your goal is to force a defense of the proposal's most fundamental assumptions. Do not offer solutions. "
                         "Your output is designed to be refuted, not integrated. Critique the following artifact. "
                         "Provide your output as a list of dictionary objects with 'severity' (always 'HIGH'), 'description' "
                         "(the counter-argument), and 'recommendation' (always 'Refute this ideological challenge.')."
    },
    "CodeAuditor": {
        "critique_type": "CONSTRUCTIVE",
        "system_prompt": "Assume the role of an expert Code Auditor. Your function is to perform a security and quality "
                         "audit of the provided source code. Your analysis checklist includes: Security Vulnerabilities "
                         "(SQL Injection, XSS, etc.), Performance Bottlenecks, Readability, Maintainability, Error Handling,"
                         " and adherence to Best Practices. Critique the following code artifact. Provide your output as a list of "
                         "dictionary objects with 'severity' ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW'), 'description', and 'recommendation' keys."
    },
    "SecurityHawk": {
        "critique_type": "CONSTRUCTIVE",
        "system_prompt": "Assume the role of a zero-trust Security Architect. Analyze the artifact strictly for IAM vulnerabilities, "
                         "data exposure risks, and non-compliance with SOC2..."
    }
}


def load_personas() -> dict:
    """Loads default personas and merges any user-defined personas from custom_personas.json."""
    personas = DEFAULT_ADVERSARY_PERSONAS.copy()
    custom_file_path = os.getenv("MASC_PERSONAS_FILE", "custom_personas.json")

    if os.path.exists(custom_file_path):
        try:
            with open(custom_file_path, 'r', encoding='utf-8') as f:
                custom_personas = json.load(f)

            # Validate the structure of the custom personas
            for name, config in custom_personas.items():
                if "critique_type" in config and "system_prompt" in config:
                    personas[name] = config
                else:
                    logger.warning(
                        f"Skipping custom persona '{name}': Missing required 'critique_type' or 'system_prompt'.")

            logger.info(f"Successfully loaded {len(custom_personas)} custom persona(s) from {custom_file_path}.")
        except Exception as e:
            logger.error(f"Failed to load custom personas from {custom_file_path}: {e}")

    return personas


# This dictionary is now dynamically populated at startup
ADVERSARY_PERSONAS = load_personas()
