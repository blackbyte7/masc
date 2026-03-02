import json
import logging
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.config.settings import settings
from src.core.engine import execute_masc_workflow
from src.core.state import MASCConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(title="MASC API", version="2.0.0")


class MASCRequest(BaseModel):
    task_description: str
    config: MASCConfig


async def stream_graph_execution(task_description: str, config: MASCConfig, thread_id: str):
    try:
        async for state_update in execute_masc_workflow(task_description, config, thread_id):
            node_name = list(state_update.keys())[0]
            node_data = state_update[node_name]

            safe_state = {}
            for key, value in node_data.items():
                if hasattr(value, "model_dump"):
                    safe_state[key] = value.model_dump()
                else:
                    safe_state[key] = value

            payload = {
                "node_executed": node_name,
                "current_turn": node_data.get("current_turn", 1),
                "state_data": safe_state
            }

            yield f"data: {json.dumps(payload)}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


@app.post("/v1/masc/execute")
async def execute_workflow(request: MASCRequest):
    if len(request.config.adversaries) > settings.max_adversaries:
        raise HTTPException(status_code=400, detail=f"Exceeded max adversaries ({settings.max_adversaries})")

    thread_id = str(uuid.uuid4())
    return StreamingResponse(
        stream_graph_execution(request.task_description, request.config, thread_id),
        media_type="text/event-stream"
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.entrypoints.api_server:app", host="0.0.0.0", port=8000)
