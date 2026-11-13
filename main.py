import argparse
import logging
import sys

from dotenv import load_dotenv

load_dotenv()

from src import __version__  # noqa: E402  (load .env before settings are instantiated)
from src.utils.logging_config import configure_logging  # noqa: E402

# Route logs to stderr so MCP stdio mode never pollutes stdout
configure_logging()


def main():
    parser = argparse.ArgumentParser(description="MASC Enterprise Framework Launcher")
    parser.add_argument(
        "--mode",
        choices=["ui", "api", "cli", "mcp"],
        default="ui",
        help="Choose which interface to launch (default: ui)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind address for ui/api modes (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None, help="Port for ui/api modes (default: 7860 / 8000)")
    parser.add_argument("--version", action="version", version=f"MASC {__version__}")

    args, unknown = parser.parse_known_args()

    if args.mode == "ui":
        from src.entrypoints.gradio_ui import fastapi_app
        from src.utils.runtime import serve

        port = args.port or 7860
        logging.info(f"🚀 Launching MASC Gradio Studio (FastAPI Mounted) on {args.host}:{port}...")
        serve(fastapi_app, host=args.host, port=port)

    elif args.mode == "api":
        from src.entrypoints.api_server import app
        from src.utils.runtime import serve

        port = args.port or 8000
        logging.info(f"🚀 Launching MASC FastAPI Server on {args.host}:{port}...")
        serve(app, host=args.host, port=port)

    elif args.mode == "cli":
        logging.info("🚀 Delegating to MASC CLI...")
        from src.entrypoints.cli import main as cli_main

        sys.argv = [sys.argv[0]] + unknown
        cli_main()

    elif args.mode == "mcp":
        logging.info("🚀 Launching MASC MCP Server for Agentic Workflows...")
        from src.entrypoints.mcp_server import run as run_mcp

        run_mcp()


if __name__ == "__main__":
    main()
