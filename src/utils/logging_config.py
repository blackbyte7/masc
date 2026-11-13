import logging
import sys

from src.config.settings import settings

_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def configure_logging(level: str | None = None) -> None:
    """
    Configures root logging once for every MASC entrypoint.

    Logs always go to stderr so that the MCP stdio transport (which owns stdout for JSON-RPC)
    is never corrupted, regardless of which entrypoint is launched.
    """
    logging.basicConfig(
        stream=sys.stderr,
        level=(level or settings.log_level).upper(),
        format=_LOG_FORMAT,
        force=True,
    )
