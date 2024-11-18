from typing import Any

import uvicorn
from absl import app, flags
from langfuse.openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient

from src.service import create_fastapi_app

COLLECTION_NAME = "duckdb_examples"


FLAGS = flags.FLAGS
flags.DEFINE_string(
    "embedding_model",
    default="text-embedding-3-small",
    help="OpenAI embedding model to use",
)
flags.DEFINE_string(
    "chat_model",
    default="gpt-4o",
    help="OpenAI chat model to use",
)
flags.DEFINE_string(
    "qdrant_host",
    default="http://localhost:6333",
    help="Qdrant host to connect to",
)


def main(argv: Any) -> None:
    del argv  # Unused

    fastapi_app = create_fastapi_app(
        qdrant=AsyncQdrantClient(FLAGS.qdrant_host),
        openai=AsyncOpenAI(),
        embeding_model=FLAGS.embedding_model,
        chat_model=FLAGS.chat_model,
        collection_name=COLLECTION_NAME,
    )
    uvicorn.run(
        app=fastapi_app,
        host="0.0.0.0",
        use_colors=True,
    )


if __name__ == "__main__":
    app.run(main)
