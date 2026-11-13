"""
Cross-platform runtime helpers.

psycopg's async driver (used by the PostgreSQL checkpointer) cannot run on Windows' default
``ProactorEventLoop``. These helpers make every entrypoint use a ``SelectorEventLoop`` on Windows.
"""

import asyncio
import sys
import warnings
from typing import Any, Awaitable, TypeVar

T = TypeVar("T")

_IS_WINDOWS = sys.platform == "win32"


def configure_event_loop_policy() -> None:
    """Installs the selector event loop policy on Windows for code paths that call ``asyncio.run`` internally."""
    if not _IS_WINDOWS:
        return
    policy_cls = getattr(asyncio, "WindowsSelectorEventLoopPolicy", None)
    if policy_cls is None:  # pragma: no cover - removed in future Python versions
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        asyncio.set_event_loop_policy(policy_cls())


def run_async(coro: Awaitable[T]) -> T:
    """``asyncio.run`` that uses a selector event loop on Windows."""
    if _IS_WINDOWS:
        if sys.version_info >= (3, 12):
            return asyncio.run(coro, loop_factory=asyncio.SelectorEventLoop)  # type: ignore[arg-type,call-arg]
        configure_event_loop_policy()
    return asyncio.run(coro)  # type: ignore[arg-type]


def uvicorn_loop_setting() -> str:
    # uvicorn treats a non-builtin "module:attr" loop setting as the loop factory itself.
    return "asyncio:SelectorEventLoop" if _IS_WINDOWS else "auto"


def serve(app: Any, port: int, host: str = "0.0.0.0") -> None:
    """Runs an ASGI app with uvicorn using a psycopg-compatible event loop and MASC logging."""
    import uvicorn

    from src.utils.logging_config import configure_logging

    configure_logging()
    uvicorn.run(app, host=host, port=port, loop=uvicorn_loop_setting(), log_config=None)
