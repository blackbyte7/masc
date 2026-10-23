import json
import logging
import os

logger = logging.getLogger(__name__)

MAX_PERSONA_PROMPT_LENGTH = 20000

DEFAULT_ADVERSARY_PERSONAS = {
    "UncertaintyQuantifier": {
        "critique_type": "CONSTRUCTIVE",
        "system_prompt": (
            "Assume the role of The Uncertainty Quantifier. Your sole function is to perform a rigorous "
            "epistemic audit of the provided text. Your analysis checklist includes: Unsubstantiated Claims, "
            "Ambiguous Terminology, Unstated Assumptions, Overconfident Extrapolation, and Neglect of Contested Knowledge. "
            "Critique the following artifact against the original task description. Provide your output as a list of "
            "critique items with 'severity' ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW'), 'description', and 'recommendation' fields."
        ),
    },
    "ContextualChallenger": {
        "critique_type": "CONSTRUCTIVE",
        "system_prompt": (
            "Assume the role of The Contextual Challenger. Your function is to analyze the provided text "
            "for its contextual blind spots. Your analysis checklist includes: Stakeholder Omissions, "
            "Ethical and Moral Blind Spots, Cross-Domain Myopia, Implementation and Practicality Barriers, "
            "and Potential for Misuse. Critique the following artifact against the original task description. Provide "
            "your output as a list of critique items with 'severity' ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW'), "
            "'description', and 'recommendation' fields."
        ),
    },
    "DA": {
        "critique_type": "ANTAGONISTIC",
        "system_prompt": (
            "Assume the role of The Devil's Advocate. Your function is not to provide constructive feedback. "
            "Your task is to be a pure antagonist. Identify the central thesis of the provided text and "
            "formulate the strongest possible counter-argument, even if that argument is specious or contrarian. "
            "Your goal is to force a defense of the proposal's most fundamental assumptions. Do not offer solutions. "
            "Your output is designed to be refuted, not integrated. Critique the following artifact. "
            "Provide your output as a list of critique items with 'severity' (always 'HIGH'), 'description' "
            "(the counter-argument), and 'recommendation' (always 'Refute this ideological challenge.')."
        ),
    },
    "CodeAuditor": {
        "critique_type": "CONSTRUCTIVE",
        "system_prompt": (
            "Assume the role of an expert Code Auditor. Your function is to perform a security and quality "
            "audit of the provided source code or technical architecture. Your analysis checklist includes: "
            "Security Vulnerabilities (SQL Injection, XSS, SSRF, etc.), Performance Bottlenecks, Readability, "
            "Maintainability, Error Handling, and adherence to Best Practices. Critique the following artifact "
            "against the original task description. Provide your output as a list of critique items with 'severity' "
            "('CRITICAL', 'HIGH', 'MEDIUM', 'LOW'), 'description', and 'recommendation' fields."
        ),
    },
    "SecurityHawk": {
        "critique_type": "CONSTRUCTIVE",
        "system_prompt": (
            "Assume the role of a zero-trust Security Architect. Analyze the artifact strictly for Identity and "
            "Access Management (IAM) vulnerabilities, data exposure risks, insecure secret handling, missing encryption "
            "in transit or at rest, and non-compliance with SOC2, ISO 27001, and OWASP Top 10 standards. "
            "Critique the following artifact against the original task description. Provide your output as a list of "
            "critique items with 'severity' ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW'), 'description', and 'recommendation' fields."
        ),
    },
}


def load_personas() -> dict:
    """Loads default personas and merges any user-defined personas from custom_personas.json."""
    personas = DEFAULT_ADVERSARY_PERSONAS.copy()
    custom_file_path = os.getenv("MASC_PERSONAS_FILE", "custom_personas.json")

    if os.path.exists(custom_file_path):
        try:
            with open(custom_file_path, "r", encoding="utf-8") as f:
                custom_personas = json.load(f)

            if not isinstance(custom_personas, dict):
                logger.warning(f"Custom personas file '{custom_file_path}' must contain a JSON object.")
                return personas

            loaded_count = 0
            for name, config in custom_personas.items():
                if (
                    isinstance(config, dict)
                    and config.get("critique_type") in ("CONSTRUCTIVE", "ANTAGONISTIC")
                    and isinstance(config.get("system_prompt"), str)
                    and 0 < len(config["system_prompt"].strip()) <= MAX_PERSONA_PROMPT_LENGTH
                ):
                    personas[name] = {
                        "critique_type": config["critique_type"],
                        "system_prompt": config["system_prompt"].strip(),
                    }
                    loaded_count += 1
                else:
                    logger.warning(
                        f"Skipping custom persona '{name}': Missing, invalid, or oversized 'critique_type' or 'system_prompt'."
                    )

            logger.info(f"Successfully loaded {loaded_count} custom persona(s) from {custom_file_path}.")
        except Exception as e:
            logger.error(f"Failed to load custom personas from {custom_file_path}: {e}")

    return personas


# Populated at startup and refreshed dynamically via load_personas()
ADVERSARY_PERSONAS = load_personas()
