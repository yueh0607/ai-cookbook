from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response, StreamingResponse
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from src.chat import chat_with_tools
from src.tools import run_sql

ALLOWED_MODELS = ["gpt-4o", "gpt-4o-mini"]


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatCompletionMessageParam]
    stream: bool = False


def create_fastapi_app(client: AsyncOpenAI) -> FastAPI:
    app = FastAPI()

    @app.get("/webui")
    async def health_check() -> JSONResponse:
        return JSONResponse(content={"status": "ok"})

    @app.get("/webui/models")
    async def list_models() -> Response:
        models = await client.models.list()
        models.data = [model for model in models.data if model.id in ALLOWED_MODELS]
        return Response(content=models.model_dump_json(), media_type="application/json")

    @app.post("/webui/chat/completions")
    async def webui_chat(request: ChatCompletionRequest) -> Any:
        # Title generation
        if not request.stream:
            completion = await client.chat.completions.create(
                model=request.model, messages=request.messages
            )
            return Response(
                content=completion.model_dump_json(), media_type="application/json"
            )

        stream = await chat_with_tools(
            client=client,
            model=request.model,
            messages=request.messages,
            tool_functions=[run_sql],
            stream=True,
        )

        async def stream_content() -> AsyncIterator[str]:
            async for chunk in stream:
                yield f"data: {chunk.model_dump_json()}\n\n"

        return StreamingResponse(
            content=stream_content(), media_type="text/event-stream"
        )

    @app.post("/v1/chat/completions")
    async def chat(request: ChatCompletionRequest) -> Any:
        return await chat_with_tools(
            client=client,
            model=request.model,
            messages=request.messages,
            tool_functions=[run_sql],
            stream=request.stream,
        )

    return app
