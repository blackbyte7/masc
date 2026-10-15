import json
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.config.settings import settings
from src.core.engine import close_pg_pool, execute_masc_workflow
from src.core.personas import ADVERSARY_PERSONAS, load_personas
from src.core.state import MASCConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup phase: The pool initializes lazily in engine.py when first needed
    yield
    # Shutdown phase: Teardown async resources
    await close_pg_pool()


app = FastAPI(title="MASC API", version="2.0.0", lifespan=lifespan)

allow_origins = settings.cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials="*" not in allow_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MASCRequest(BaseModel):
    task_description: str = Field(min_length=1, description="The prompt or goal for the MASC workflow.")
    config: MASCConfig


async def verify_api_key(authorization: Optional[str] = Header(default=None)):
    """Enforces Bearer token authentication when MASC_API_KEY is configured."""
    if not settings.masc_api_key:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization Bearer token.",
        )
    token = authorization.removeprefix("Bearer ").strip()
    if token != settings.masc_api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )


async def stream_graph_execution(task_description: str, config: MASCConfig, thread_id: str):
    try:
        async for state_update in execute_masc_workflow(task_description, config, thread_id):
            node_name = list(state_update.keys())[0]
            node_data = state_update[node_name]

            safe_state = {}
            for key, value in node_data.items():
                if hasattr(value, "redacted_dump"):
                    safe_state[key] = value.redacted_dump()
                elif hasattr(value, "model_dump"):
                    safe_state[key] = value.model_dump()
                else:
                    safe_state[key] = value

            raw_turn = node_data.get("current_turn", 1)
            executed_turn = (
                raw_turn - 1
                if node_name in ("synthesize_sequential", "synthesize_architect") and raw_turn > 1
                else raw_turn
            )

            payload = {
                "thread_id": thread_id,
                "node_executed": node_name,
                "current_turn": executed_turn,
                "state_data": safe_state,
            }

            yield f"data: {json.dumps(payload)}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.exception(f"Workflow execution error on thread {thread_id}: {e}")
        yield f"data: {json.dumps({'error': 'Workflow execution failed. Check server logs for details.'})}\n\n"


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/v1/masc/execute", dependencies=[Depends(verify_api_key)])
async def execute_workflow(request: MASCRequest):
    if len(request.config.adversaries) > settings.max_adversaries:
        raise HTTPException(
            status_code=400,
            detail=f"Exceeded max adversaries ({settings.max_adversaries})",
        )

    if request.config.max_turns > settings.max_turns_limit:
        raise HTTPException(
            status_code=400,
            detail=f"Exceeded max turns limit ({settings.max_turns_limit})",
        )

    available_personas = {**ADVERSARY_PERSONAS, **load_personas()}
    invalid_personas = [
        adv.persona_name
        for adv in request.config.adversaries
        if adv.persona_name not in available_personas
    ]
    if invalid_personas:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown adversary persona(s): {invalid_personas}",
        )

    thread_id = str(uuid.uuid4())
    return StreamingResponse(
        stream_graph_execution(request.task_description, request.config, thread_id),
        media_type="text/event-stream",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.entrypoints.api_server:app", host="0.0.0.0", port=8000)
