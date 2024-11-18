from typing import Any

import uvicorn
from absl import app
from langfuse.openai import AsyncOpenAI

from src.service import create_fastapi_app


def main(argv: Any) -> None:
    del argv  # Unused

    client = AsyncOpenAI()
    fastapi_app = create_fastapi_app(client=client)
    uvicorn.run(
        app=fastapi_app,
        host="0.0.0.0",
        use_colors=True,
    )


if __name__ == "__main__":
    app.run(main)
