import json
import logging
from collections import deque
from typing import Any, Callable

from langfuse.decorators import observe
from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
)
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)

from src.tools import get_tool_param

SYSTEM_PROMPT = """
You are an expert in analyzing tabular data with DuckDB SQL.

You are given access to a Hacker News table `posts`. Hacker News is a social news website focusing on computer science and entrepreneurship. Users can post stories (such as links to articles), comment on them, and vote them up or down, affecting their visibility. The table has the following schema:

```
CREATE TABLE posts (
    title VARCHAR,
    text VARCHAR,
    time BIGINT,
    timestamp TIMESTAMP,
    votes INTEGER,
    comments INTEGER
)
```
"""


def execute_tool_call(
    tool_call: ChatCompletionMessageToolCall,
    function_map: dict[str, Callable[..., Any]],
) -> ChatCompletionToolMessageParam:
    function_name: str = tool_call.function.name
    if function_name not in function_map:
        return ChatCompletionToolMessageParam(
            content=f"Function {function_name} not found",
            role="tool",
            tool_call_id=tool_call.id,
        )

    function: Callable[..., Any] = function_map[function_name]
    function_args: dict[str, Any] = json.loads(tool_call.function.arguments)

    logging.info(f"Calling function {function_name} with args {function_args}")
    function_response: Any = function(**function_args)
    return ChatCompletionToolMessageParam(
        content=str(function_response),
        role="tool",
        tool_call_id=tool_call.id,
    )


@observe()
async def chat_with_tools(
    client: AsyncOpenAI,
    model: str,
    messages: list[ChatCompletionMessageParam],
    tool_functions: list[Callable[..., Any]],
    stream: bool,
) -> ChatCompletion | AsyncStream[ChatCompletionChunk]:
    tools = [get_tool_param(agent_function) for agent_function in tool_functions]
    function_map = {f.__name__: f for f in tool_functions}

    tool_calls = deque()

    messages = [
        ChatCompletionSystemMessageParam(content=SYSTEM_PROMPT, role="system"),
        *messages,
    ]

    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
    )
    response_message = response.choices[0].message

    # Extend conversation with assistant response
    messages.append(response_message)

    for tool_call in response_message.tool_calls:
        tool_calls.append(tool_call)

    while tool_calls:
        tool_call = tool_calls.popleft()

        try:
            tool_call_message = execute_tool_call(
                tool_call=tool_call, function_map=function_map
            )
            messages.append(tool_call_message)
        except Exception as e:
            tool_call_message = ChatCompletionToolMessageParam(
                content=str(e),
                role="tool",
                tool_call_id=tool_call.id,
            )
            messages.append(tool_call_message)

            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
            )
            response_message = response.choices[0].message
            messages.append(response_message)
            for tool_call in response_message.tool_calls:
                tool_calls.append(tool_call)

    return await client.chat.completions.create(
        model=model, messages=messages, stream=stream
    )
