import asyncio
import inspect
import json
import logging
import secrets
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.config.settings import settings, validate_base_url
from src.core.engine import (
    close_pg_pool,
    execute_masc_workflow,
    get_thread_checkpoint_state,
)
from src.core.personas import ADVERSARY_PERSONAS, load_personas, resolve_adversary_persona
from src.core.state import MASCConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup phase: The pool initializes lazily in engine.py when first needed
    yield
    # Shutdown phase: Teardown async resources
    await close_pg_pool()


app = FastAPI(title="MASC API", version="2.2.0", lifespan=lifespan)

allow_origins = settings.cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials="*" not in allow_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MASCRequest(BaseModel):
    task_description: str = Field(
        min_length=1,
        max_length=settings.max_task_length,
        description="The prompt or goal for the MASC workflow.",
    )
    config: MASCConfig
    initial_artifact: Optional[str] = Field(
        default=None,
        max_length=settings.max_task_length,
        description="Optional existing text/code artifact to audit directly without running the Proposer node.",
    )
    thread_id: Optional[str] = Field(
        default=None,
        description="Optional thread ID for correlating or inspecting checkpointed execution state.",
    )


async def verify_api_key(authorization: Optional[str] = Header(default=None)):
    """Enforces constant-time Bearer token authentication when MASC_API_KEY is configured."""
    if not settings.masc_api_key:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization Bearer token.",
        )
    token = authorization.removeprefix("Bearer ").strip()
    if not secrets.compare_digest(token.encode("utf-8"), settings.masc_api_key.encode("utf-8")):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )


def _validate_config_security(config: MASCConfig):
    """Validates all persona LLM configurations for SSRF and API key exfiltration risks."""
    all_personas = [config.proposer, config.synthesizer, *config.adversaries]
    for persona in all_personas:
        llm_cfg = persona.llm_config
        try:
            validated_url = validate_base_url(llm_cfg.base_url, settings)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        raw_key = (llm_cfg.api_key or "").strip()
        if validated_url and llm_cfg.provider != "Ollama" and raw_key in ("", "env"):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Explicit API key is required when specifying a custom base_url "
                    f"for provider '{llm_cfg.provider}'."
                ),
            )


def _invoke_workflow_generator(
    task_description: str,
    config: MASCConfig,
    thread_id: str,
    initial_artifact: Optional[str] = None,
):
    """Calls execute_masc_workflow while remaining compatible with 3-arg test monkeypatches."""
    sig = inspect.signature(execute_masc_workflow)
    if "initial_artifact" in sig.parameters:
        return execute_masc_workflow(
            task_description,
            config,
            thread_id,
            initial_artifact=initial_artifact,
        )
    return execute_masc_workflow(task_description, config, thread_id)


async def stream_graph_execution(
    task_description: str,
    config: MASCConfig,
    thread_id: str,
    http_request: Optional[Request] = None,
    initial_artifact: Optional[str] = None,
):
    queue: asyncio.Queue[Optional[str]] = asyncio.Queue()

    async def _producer():
        try:
            async for state_update in _invoke_workflow_generator(
                task_description, config, thread_id, initial_artifact=initial_artifact
            ):
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

                await queue.put(f"data: {json.dumps(payload)}\n\n")
            await queue.put("data: [DONE]\n\n")
        except asyncio.CancelledError:
            logger.info(f"Client disconnected; cancelled workflow on thread {thread_id}.")
            raise
        except Exception as e:
            logger.exception(f"Workflow execution error on thread {thread_id}: {e}")
            await queue.put(
                f"data: {json.dumps({'error': 'Workflow execution failed. Check server logs for details.'})}\n\n"
            )
        finally:
            await queue.put(None)

    producer_task = asyncio.create_task(_producer())
    try:
        while True:
            if http_request is not None and await http_request.is_disconnected():
                producer_task.cancel()
                break
            try:
                item = await asyncio.wait_for(queue.get(), timeout=settings.sse_heartbeat_interval)
                if item is None:
                    break
                yield item
            except asyncio.TimeoutError:
                # Emit SSE comment heartbeat to keep reverse proxies and load balancers alive
                yield ": keep-alive\n\n"
    finally:
        if not producer_task.done():
            producer_task.cancel()


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/v1/masc/threads/{thread_id}", dependencies=[Depends(verify_api_key)])
async def get_thread_state(thread_id: str):
    state = await get_thread_checkpoint_state(thread_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Thread '{thread_id}' not found.")
    return state


@app.post("/v1/masc/execute", dependencies=[Depends(verify_api_key)])
async def execute_workflow(masc_request: MASCRequest, request: Request):
    if len(masc_request.config.adversaries) > settings.max_adversaries:
        raise HTTPException(
            status_code=400,
            detail=f"Exceeded max adversaries ({settings.max_adversaries})",
        )

    if masc_request.config.max_turns > settings.max_turns_limit:
        raise HTTPException(
            status_code=400,
            detail=f"Exceeded max turns limit ({settings.max_turns_limit})",
        )

    available_personas = {**ADVERSARY_PERSONAS, **load_personas()}
    invalid_personas = []
    for adv in masc_request.config.adversaries:
        try:
            resolve_adversary_persona(adv, available_personas)
        except ValueError:
            invalid_personas.append(adv.persona_name)

    if invalid_personas:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown adversary persona(s): {invalid_personas}",
        )

    _validate_config_security(masc_request.config)

    thread_id = masc_request.thread_id or str(uuid.uuid4())
    return StreamingResponse(
        stream_graph_execution(
            masc_request.task_description,
            masc_request.config,
            thread_id,
            request,
            initial_artifact=masc_request.initial_artifact,
        ),
        media_type="text/event-stream",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.entrypoints.api_server:app", host="0.0.0.0", port=8000)
