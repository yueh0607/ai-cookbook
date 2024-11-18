import json
import logging
from typing import Any, Callable

from langfuse.decorators import observe
from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)

from src.tools import get_tool_param

SYSTEM_PROMPT = """
You are an expert in analyzing tabular data with DuckDB SQL.

You have access to a Hacker News table called `posts`. Hacker News is a social news website centered on computer science and entrepreneurship, where users can post stories (e.g., links to articles), comment on them, and upvote or downvote to influence visibility. The table has the following schema:

```sql
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

THINK_PROMPT = """
Generate DuckDB SQL for the following query.

**Query**: {query}

Think step by step and show SQL for each step. 
- Prefer WITH statements over subqueries. 
- Prefer solutions that don't use joins.
- Review the final SQL and simplify it if possible. Only keep the relevant columns.
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

    try:
        logging.info(f"Calling function {function_name} with args {function_args}")
        function_response: Any = function(**function_args)
        return ChatCompletionToolMessageParam(
            content=str(function_response),
            role="tool",
            tool_call_id=tool_call.id,
        )
    except Exception as e:
        return ChatCompletionToolMessageParam(
            content=str(e),
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
    query = messages[-1]["content"]

    # Think step with rewritten query
    messages[-1] = ChatCompletionUserMessageParam(
        content=THINK_PROMPT.format(query=query),
        role="user",
    )

    response = await client.chat.completions.create(
        model=model,
        messages=[
            ChatCompletionSystemMessageParam(content=SYSTEM_PROMPT, role="system"),
            *messages,
        ],
    )
    response_message = response.choices[0].message
    messages.append(response_message)

    # Call with tools and the original query
    messages.append(ChatCompletionUserMessageParam(content=query, role="user"))
    tools = [get_tool_param(agent_function) for agent_function in tool_functions]
    response = await client.chat.completions.create(
        model=model,
        messages=[
            ChatCompletionSystemMessageParam(content=SYSTEM_PROMPT, role="system"),
            *messages,
        ],
        tools=tools,
    )
    response_message = response.choices[0].message
    messages.append(response_message)

    tool_calls = response_message.tool_calls or []
    function_map = {f.__name__: f for f in tool_functions}

    # Add tool calls results to messages.
    # Use for loop because there could be multiple calls for the same tool
    for tool_call in tool_calls:
        messages.append(
            execute_tool_call(tool_call=tool_call, function_map=function_map)
        )

    return await client.chat.completions.create(
        model=model, messages=messages, stream=stream
    )
