import ast
import json
import logging
import os
import re
from typing import List, Optional

from src.core.state import PersonaConfig

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

_CODE_BLOCK_RE = re.compile(r"```(?P<lang>[a-zA-Z0-9_+-]*)\n(?P<code>.*?)```", re.DOTALL)
_SECRET_PATTERNS = [
    ("OpenAI API Key pattern", re.compile(r"sk-[a-zA-Z0-9]{20,}")),
    ("AWS Access Key ID pattern", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("PEM Private Key header", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
]
_DANGEROUS_CALLS = {"eval", "exec"}


def run_deterministic_verification(persona_name: str, content: str) -> List[str]:
    """
    Runs lightweight, deterministic static checks on the artifact to ground adversary critiques
    with concrete verification findings (AST syntax, dangerous calls, secret patterns, JSON validity).
    """
    findings: List[str] = []
    if not content or not content.strip():
        return findings

    logger.debug(f"Running deterministic static verification for adversary '{persona_name}'.")

    # 1. Check for hardcoded secret patterns
    for label, pattern in _SECRET_PATTERNS:
        if pattern.search(content):
            findings.append(f"[STATIC SECRET SCAN] Detected potential hardcoded credential matching {label}.")

    # 2. Extract fenced code blocks for Python AST & JSON verification
    blocks = list(_CODE_BLOCK_RE.finditer(content))
    for idx, match in enumerate(blocks, start=1):
        lang = (match.group("lang") or "").strip().lower()
        code = match.group("code")

        if lang in ("python", "py"):
            try:
                tree = ast.parse(code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call):
                        func_name = getattr(node.func, "id", None)
                        attr_name = getattr(node.func, "attr", None)
                        if func_name in _DANGEROUS_CALLS:
                            findings.append(
                                f"[AST AUDIT] Python block #{idx} invokes dangerous builtin '{func_name}()' "
                                f"at line {getattr(node, 'lineno', '?')}."
                            )
                        if attr_name in ("loads", "load") and getattr(getattr(node.func, "value", None), "id", "") in (
                            "pickle",
                            "yaml",
                        ):
                            findings.append(
                                f"[AST AUDIT] Python block #{idx} uses potentially unsafe deserialization "
                                f"at line {getattr(node, 'lineno', '?')}."
                            )
                        for kw in node.keywords:
                            if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                                findings.append(
                                    f"[AST AUDIT] Python block #{idx} enables 'shell=True' command execution "
                                    f"at line {getattr(node, 'lineno', '?')}."
                                )
            except SyntaxError as exc:
                findings.append(
                    f"[AST SYNTAX ERROR] Python code block #{idx} failed to parse: {exc.msg} (line {exc.lineno})."
                )

        elif lang == "json":
            try:
                json.loads(code)
            except json.JSONDecodeError as exc:
                findings.append(f"[JSON SYNTAX ERROR] JSON block #{idx} is invalid: {exc.msg} (line {exc.lineno}).")

    return findings


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


def resolve_adversary_persona(adv_config: PersonaConfig, registry: Optional[dict] = None) -> dict:
    """
    Resolves an adversary's persona definition either from inline fields on PersonaConfig
    or from the global/custom persona registry.
    """
    if adv_config.critique_type and adv_config.system_prompt and adv_config.system_prompt.strip():
        return {
            "critique_type": adv_config.critique_type,
            "system_prompt": adv_config.system_prompt.strip(),
        }

    personas_map = registry if registry is not None else {**ADVERSARY_PERSONAS, **load_personas()}
    if adv_config.persona_name in personas_map:
        return personas_map[adv_config.persona_name]

    raise ValueError(
        f"Unknown adversary persona '{adv_config.persona_name}'. Either select a registered persona "
        f"({list(personas_map.keys())}) or provide inline 'critique_type' and 'system_prompt'."
    )


# Populated at startup and refreshed dynamically via load_personas()
ADVERSARY_PERSONAS = load_personas()
