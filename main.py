import argparse
import logging
import sys

from dotenv import load_dotenv

load_dotenv()
# Route logs to stderr so MCP stdio mode never pollutes stdout
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def main():
    parser = argparse.ArgumentParser(description="MASC Enterprise Framework Launcher")
    parser.add_argument(
        "--mode",
        choices=["ui", "api", "cli", "mcp"],
        default="ui",
        help="Choose which interface to launch (default: ui)",
    )

    args, unknown = parser.parse_known_args()

    if args.mode == "ui":
        logging.info("🚀 Launching MASC Gradio Studio (FastAPI Mounted) on port 7860...")
        import uvicorn
        from src.entrypoints.gradio_ui import fastapi_app

        uvicorn.run(fastapi_app, host="0.0.0.0", port=7860)

    elif args.mode == "api":
        logging.info("🚀 Launching MASC FastAPI Server on port 8000...")
        import uvicorn

        uvicorn.run("src.entrypoints.api_server:app", host="0.0.0.0", port=8000)

    elif args.mode == "cli":
        logging.info("🚀 Delegating to MASC CLI...")
        from src.entrypoints.cli import main as cli_main

        sys.argv = [sys.argv[0]] + unknown
        cli_main()

    elif args.mode == "mcp":
        logging.info("🚀 Launching MASC MCP Server for Agentic Workflows...")
        from src.entrypoints.mcp_server import mcp

        mcp.run()


if __name__ == "__main__":
    main()
