import json
import logging
from collections import deque
from dataclasses import dataclass
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
from qdrant_client import AsyncQdrantClient

from src.tools import get_tool_param

SYSTEM_PROMPT = """
You are an expert in analyzing tabular data with DuckDB SQL.

You are given access to a Hacker News table `posts`. Hacker News is a social news website focusing on computer science and entrepreneurship. Users can post stories (such as links to articles), comment on them, and vote them up or down, affecting their visibility. The table has the following schema:

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

Below are some example DuckDB SQL queries:

{context}
"""

EXAMPLE_TEMPLATE = """
### Task: {task}

```sql
{sql}
```

Explanation: {explanation}
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


@dataclass
class Example:
    task: str
    sql: str
    explanation: str
    score: float


class ChatManager:
    def __init__(
        self,
        qdrant: AsyncQdrantClient,
        openai: AsyncOpenAI,
        embedding_model: str = "text-embedding-3-small",
        chat_model: str = "gpt-4o",
    ) -> None:
        self.qdrant = qdrant
        self.openai = openai
        self.embedding_model = embedding_model
        self.chat_model = chat_model

    @observe()
    async def search(
        self,
        query: str,
        collection_name: str,
        top_k: int = 10,
        # score_threshold: float = 0.2,
    ) -> list[Example]:
        """Search for similar documents in the Qdrant index."""

        embedding_response = await self.openai.embeddings.create(
            input=query, model=self.embedding_model
        )
        embedding = embedding_response.data[0].embedding

        search_response = await self.qdrant.search(
            collection_name=collection_name,
            query_vector=embedding,
            limit=top_k,
            with_payload=True,
            # score_threshold=score_threshold,
        )

        examples = [
            Example(
                task=point.payload.get("task", ""),
                sql=point.payload.get("sql", ""),
                explanation=point.payload.get("explanation", ""),
                score=point.score,
            )
            for point in search_response
            if point.payload
        ]
        logging.info(f"Found {len(examples)} examples related to the query")
        return examples

    @observe()
    async def chat_with_tools(
        self,
        messages: list[ChatCompletionMessageParam],
        collection_name: str,
        tool_functions: list[Callable[..., Any]],
        stream: bool,
    ) -> ChatCompletion | AsyncStream[ChatCompletionChunk]:
        # Add few-shot examples to the prompt
        query = "\n".join(
            [message["content"] for message in messages if message["role"] == "user"]
        )
        logging.info(f"Finding few-shot examples related to the query: {query}")

        examples = await self.search(query=query, collection_name=collection_name)
        context = "\n".join(
            [
                EXAMPLE_TEMPLATE.format(
                    task=example.task,
                    sql=example.sql,
                    explanation=example.explanation,
                )
                for example in examples
            ]
        )

        messages = [
            ChatCompletionSystemMessageParam(
                content=SYSTEM_PROMPT.format(context=context), role="system"
            ),
            *messages,
        ]

        # Prepare tools for the chat completion
        tools = [get_tool_param(agent_function) for agent_function in tool_functions]
        function_map = {f.__name__: f for f in tool_functions}

        tool_calls = deque()

        response = await self.openai.chat.completions.create(
            model=self.chat_model,
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

                response = await self.openai.chat.completions.create(
                    model=self.chat_model,
                    messages=messages,
                    tools=tools,
                )
                response_message = response.choices[0].message
                messages.append(response_message)
                for tool_call in response_message.tool_calls:
                    tool_calls.append(tool_call)

        return await self.openai.chat.completions.create(
            model=self.chat_model, messages=messages, stream=stream
        )
