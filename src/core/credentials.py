"""
Ephemeral, process-local credential vault.

User-supplied LLM API keys must never be written to the LangGraph checkpointer (PostgreSQL or the
in-memory saver), because checkpoints are long-lived and readable by anyone with database access.

Before a workflow starts, every non-empty ``api_key`` in the ``MASCConfig`` is replaced by an opaque
reference (``masc-vault:<random>``) and the real secret is kept only in this bounded in-process vault,
scoped to the workflow's ``thread_id``. Nodes resolve the reference just before instantiating the
LLM client. References naturally survive ``model_copy`` (e.g. when the Auto-Persona Router clones the
synthesizer's ``llm_config`` for new adversaries).

If the process restarts between a Human-in-the-Loop pause and its resume, the vault is empty and the
reference resolves to ``None``; the provider SDK then falls back to the server's environment keys,
which is the same behaviour as leaving the key blank. The custom ``base_url`` exfiltration guard in
``MASCNodes.get_llm`` still applies.
"""

import logging
import secrets
import threading
from collections import OrderedDict
from typing import Optional

from src.core.state import MASCConfig

logger = logging.getLogger(__name__)

VAULT_REF_PREFIX = "masc-vault:"
_PLACEHOLDER_KEYS = ("", "env")


class CredentialVault:
    """Thread-safe, size-bounded mapping of ``thread_id -> {reference: secret}``."""

    def __init__(self, max_threads: int = 1000) -> None:
        self._max_threads = max_threads
        self._store: OrderedDict[str, dict[str, str]] = OrderedDict()
        self._lock = threading.Lock()

    def put(self, thread_id: str, secret: str) -> str:
        ref = f"{VAULT_REF_PREFIX}{secrets.token_urlsafe(16)}"
        with self._lock:
            bucket = self._store.setdefault(thread_id, {})
            bucket[ref] = secret
            self._store.move_to_end(thread_id)
            while len(self._store) > self._max_threads:
                self._store.popitem(last=False)
        return ref

    def get(self, thread_id: Optional[str], ref: str) -> Optional[str]:
        if not thread_id:
            return None
        with self._lock:
            bucket = self._store.get(thread_id)
            if bucket is None:
                return None
            self._store.move_to_end(thread_id)
            return bucket.get(ref)

    def forget(self, thread_id: str) -> None:
        with self._lock:
            self._store.pop(thread_id, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


vault = CredentialVault()


def is_vault_reference(api_key: Optional[str]) -> bool:
    return bool(api_key) and api_key.startswith(VAULT_REF_PREFIX)  # type: ignore[union-attr]


def seal_config_credentials(config: MASCConfig, thread_id: str) -> MASCConfig:
    """
    Returns a deep copy of ``config`` whose API keys are replaced by vault references.
    The original config object is left untouched.
    """
    sealed = config.model_copy(deep=True)
    for persona in (sealed.proposer, sealed.synthesizer, *sealed.adversaries):
        # deepcopy preserves aliasing, so personas built from one shared LLMConfig would still share it.
        # Give every persona its own copy so sealing one never affects another.
        persona.llm_config = persona.llm_config.model_copy()
        raw_key = (persona.llm_config.api_key or "").strip()
        if is_vault_reference(raw_key):
            # Externally supplied vault references are never honoured: they could be replayed to borrow
            # another thread's secret. Treat them as "no key supplied".
            logger.warning("Ignoring externally supplied vault reference for persona '%s'.", persona.persona_name)
            persona.llm_config.api_key = ""
            continue
        if raw_key in _PLACEHOLDER_KEYS:
            continue
        persona.llm_config.api_key = vault.put(thread_id, raw_key)
    return sealed


def resolve_api_key(raw_key: Optional[str], thread_id: Optional[str]) -> Optional[str]:
    """
    Resolves an ``api_key`` field to the real secret.

    Returns ``None`` for blank / ``"env"`` placeholders and for vault references that are no longer
    available, so that provider SDKs fall back to environment variables.
    """
    key = (raw_key or "").strip()
    if key in _PLACEHOLDER_KEYS:
        return None
    if is_vault_reference(key):
        secret = vault.get(thread_id, key)
        if secret is None:
            logger.warning(
                "Ephemeral API key for thread '%s' is no longer available (process restart or eviction); "
                "falling back to server environment credentials.",
                thread_id,
            )
        return secret
    return key
