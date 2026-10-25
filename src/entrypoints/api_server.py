import asyncio
import inspect
import json
import logging
import secrets
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Any, List, Literal, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.config.settings import settings, validate_base_url
from src.core.engine import (
    close_pg_pool,
    execute_masc_workflow,
    get_thread_checkpoint_state,
    resume_masc_workflow,
)
from src.core.personas import ADVERSARY_PERSONAS, load_personas, resolve_adversary_persona
from src.core.state import CritiquesCollection, LLMConfig, MASCConfig, PersonaConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Sliding-window rate limit state: client_ip -> deque[timestamps]
_rate_limit_buckets: dict[str, deque[float]] = defaultdict(deque)

# Preset virtual models exposed by the OpenAI-compatible /v1/models and /v1/chat/completions endpoints
OPENAI_VIRTUAL_MODELS = {
    "masc-sequential-default": {
        "synthesis_protocol": "Sequential Refinement",
        "adversaries": ["DA", "UncertaintyQuantifier"],
        "auto_select_adversaries": False,
        "enable_verification_tools": False,
    },
    "masc-architect-default": {
        "synthesis_protocol": "Architect",
        "adversaries": ["DA", "ContextualChallenger"],
        "auto_select_adversaries": False,
        "enable_verification_tools": False,
    },
    "masc-security-audit": {
        "synthesis_protocol": "Sequential Refinement",
        "adversaries": ["SecurityHawk", "CodeAuditor"],
        "auto_select_adversaries": False,
        "enable_verification_tools": True,
    },
    "masc-auto": {
        "synthesis_protocol": "Sequential Refinement",
        "adversaries": [],
        "auto_select_adversaries": True,
        "enable_verification_tools": True,
    },
}


def reset_rate_limit_buckets() -> None:
    """Clears the in-memory sliding-window rate limit buckets."""
    _rate_limit_buckets.clear()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Startup phase: The pool initializes lazily in engine.py when first needed
    yield
    # Shutdown phase: Teardown async resources
    await close_pg_pool()


app = FastAPI(title="MASC API", version="2.4.0", lifespan=lifespan)

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


class MASCResumeRequest(BaseModel):
    approve: bool = Field(
        default=True,
        description="If True, resumes synthesis; if False, aborts the paused workflow.",
    )
    critiques_collection: Optional[CritiquesCollection] = Field(
        default=None,
        description="Optional edited/filtered CritiquesCollection to replace the checkpointed critiques before synthesis.",
    )


class OpenAIChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "developer"] = "user"
    content: str


class OpenAIStreamOptions(BaseModel):
    include_usage: bool = False


class OpenAIChatCompletionRequest(BaseModel):
    model: str = Field(default="masc-sequential-default")
    messages: List[OpenAIChatMessage] = Field(min_length=1)
    stream: bool = Field(default=False)
    stream_options: Optional[OpenAIStreamOptions] = Field(default=None)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    masc_config: Optional[MASCConfig] = Field(
        default=None,
        description="Optional full MASCConfig override for power users calling the OpenAI-compatible endpoint.",
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


def _extract_client_ip(request: Request) -> str:
    """Extracts client IP, respecting X-Forwarded-For when deployed behind a reverse proxy."""
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


async def verify_rate_limit(request: Request):
    """Enforces a sliding-window per-IP request rate limit and prunes expired IP buckets."""
    limit = settings.rate_limit_per_minute
    if limit <= 0:
        return

    client_ip = _extract_client_ip(request)
    now = time.monotonic()
    window_start = now - 60.0

    # Evict expired timestamps across tracked buckets so _rate_limit_buckets never leaks memory
    stale_ips = []
    for ip, ip_bucket in list(_rate_limit_buckets.items()):
        while ip_bucket and ip_bucket[0] < window_start:
            ip_bucket.popleft()
        if not ip_bucket and ip != client_ip:
            stale_ips.append(ip)
    for ip in stale_ips:
        _rate_limit_buckets.pop(ip, None)

    bucket = _rate_limit_buckets[client_ip]
    if len(bucket) >= limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded ({limit} requests per minute).",
        )
    bucket.append(now)


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
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if "initial_artifact" in sig.parameters or has_var_kw:
        return execute_masc_workflow(
            task_description,
            config,
            thread_id,
            initial_artifact=initial_artifact,
        )
    return execute_masc_workflow(task_description, config, thread_id)


async def _stream_events_from_generator(
    workflow_gen: Any,
    thread_id: str,
    http_request: Optional[Request] = None,
):
    queue: asyncio.Queue[Optional[str]] = asyncio.Queue()

    async def _producer():
        try:
            async for state_update in workflow_gen:
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


async def stream_graph_execution(
    task_description: str,
    config: MASCConfig,
    thread_id: str,
    http_request: Optional[Request] = None,
    initial_artifact: Optional[str] = None,
):
    gen = _invoke_workflow_generator(
        task_description, config, thread_id, initial_artifact=initial_artifact
    )
    async for chunk in _stream_events_from_generator(gen, thread_id, http_request):
        yield chunk


def _build_config_from_openai_request(req: OpenAIChatCompletionRequest) -> MASCConfig:
    if req.masc_config is not None:
        cfg = req.masc_config.model_copy(deep=True)
        # Never allow stateless OpenAI chat completions to hang on an unresumable HITL interrupt
        cfg.require_human_approval = False
        return cfg

    preset = OPENAI_VIRTUAL_MODELS.get(req.model, OPENAI_VIRTUAL_MODELS["masc-sequential-default"])
    default_llm = LLMConfig(
        provider=settings.default_provider,
        api_key="",
        model_name=settings.default_model,
        temperature=req.temperature if req.temperature is not None else settings.default_temperature,
        max_tokens=req.max_tokens if req.max_tokens is not None else settings.default_max_tokens,
    )
    adv_cfgs = [
        PersonaConfig(persona_name=adv_name, llm_config=default_llm)
        for adv_name in preset["adversaries"]
    ]
    return MASCConfig(
        proposer=PersonaConfig(persona_name="Proposer", llm_config=default_llm),
        synthesizer=PersonaConfig(persona_name="Synthesizer", llm_config=default_llm),
        adversaries=adv_cfgs,
        auto_select_adversaries=preset["auto_select_adversaries"],
        synthesis_protocol=preset["synthesis_protocol"],
        enable_verification_tools=preset["enable_verification_tools"],
        require_human_approval=False,
        max_turns=1,
    )


async def _stream_openai_completion_chunks(
    task_description: str,
    config: MASCConfig,
    thread_id: str,
    created_ts: int,
    req: OpenAIChatCompletionRequest,
):
    """Streams OpenAI-compatible chat.completion.chunk SSE events."""
    role_chunk = {
        "id": thread_id,
        "object": "chat.completion.chunk",
        "created": created_ts,
        "model": req.model,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(role_chunk)}\n\n"

    synthesized_text = ""
    latest_usage = None
    async for step_state in _invoke_workflow_generator(task_description, config, thread_id):
        step_node = list(step_state.keys())[0]
        step_data = step_state[step_node]
        if "token_usage" in step_data and step_data["token_usage"]:
            latest_usage = step_data["token_usage"]
        if step_node in ("synthesize_sequential", "synthesize_architect") and "current_artifact" in step_data:
            synthesized_text = step_data["current_artifact"].content

    content_chunk: dict[str, Any] = {
        "id": thread_id,
        "object": "chat.completion.chunk",
        "created": created_ts,
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": synthesized_text},
                "finish_reason": "stop",
            }
        ],
    }
    if req.stream_options and req.stream_options.include_usage and latest_usage:
        content_chunk["usage"] = {
            "prompt_tokens": latest_usage.input_tokens,
            "completion_tokens": latest_usage.output_tokens,
            "total_tokens": latest_usage.total_tokens,
        }

    yield f"data: {json.dumps(content_chunk)}\n\n"
    yield "data: [DONE]\n\n"


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/v1/models", dependencies=[Depends(verify_api_key)])
async def list_openai_models():
    """OpenAI-compatible model listing endpoint for IDE and UI integrations."""
    created_ts = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": created_ts,
                "owned_by": "masc-dialectics",
            }
            for model_id in OPENAI_VIRTUAL_MODELS
        ],
    }


@app.post(
    "/v1/chat/completions",
    dependencies=[Depends(verify_api_key), Depends(verify_rate_limit)],
)
async def openai_chat_completions(req: OpenAIChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint wrapping the MASC dialectical engine."""
    config = _build_config_from_openai_request(req)
    _validate_config_security(config)

    task_parts = [f"[{m.role.upper()}]: {m.content}" for m in req.messages if m.content.strip()]
    task_description = "\n\n".join(task_parts) if len(req.messages) > 1 else req.messages[-1].content
    if not task_description.strip():
        raise HTTPException(status_code=422, detail="Message content cannot be empty.")

    thread_id = f"chatcmpl-{uuid.uuid4().hex}"
    created_ts = int(time.time())

    if not req.stream:
        final_content = ""
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0

        async for state in _invoke_workflow_generator(task_description, config, thread_id):
            node_name = list(state.keys())[0]
            node_data = state[node_name]
            if "current_artifact" in node_data and node_data["current_artifact"]:
                final_content = node_data["current_artifact"].content
            usage = node_data.get("token_usage")
            if usage:
                prompt_tokens = usage.input_tokens
                completion_tokens = usage.output_tokens
                total_tokens = usage.total_tokens

        return {
            "id": thread_id,
            "object": "chat.completion",
            "created": created_ts,
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": final_content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        }

    return StreamingResponse(
        _stream_openai_completion_chunks(task_description, config, thread_id, created_ts, req),
        media_type="text/event-stream",
    )


@app.get("/v1/masc/threads/{thread_id}", dependencies=[Depends(verify_api_key)])
async def get_thread_state(thread_id: str):
    state = await get_thread_checkpoint_state(thread_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Thread '{thread_id}' not found.")
    return state


@app.post(
    "/v1/masc/threads/{thread_id}/resume",
    dependencies=[Depends(verify_api_key), Depends(verify_rate_limit)],
)
async def resume_thread_workflow(
    thread_id: str,
    resume_request: MASCResumeRequest,
    request: Request,
):
    """Resumes a paused Human-in-the-Loop MASC workflow after human inspection or critique editing."""
    state = await get_thread_checkpoint_state(thread_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Thread '{thread_id}' not found.")

    gen = resume_masc_workflow(
        thread_id=thread_id,
        critiques_override=resume_request.critiques_collection,
        approve=resume_request.approve,
    )
    return StreamingResponse(
        _stream_events_from_generator(gen, thread_id, request),
        media_type="text/event-stream",
    )


@app.post(
    "/v1/masc/execute",
    dependencies=[Depends(verify_api_key), Depends(verify_rate_limit)],
)
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
