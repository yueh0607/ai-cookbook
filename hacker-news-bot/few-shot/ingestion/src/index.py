import logging

import numpy as np
import ray
from absl import app, flags
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

INPUT_FILE = "data/examples.jsonl"
COLLECTION_NAME = "duckdb_examples"

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "embedding_model",
    default="text-embedding-3-small",
    help="OpenAI embedding model to use",
)
flags.DEFINE_integer(
    "embedding_dim",
    default=1536,
    help="Embedding dimension",
)
flags.DEFINE_string(
    "qdrant_host",
    default="http://localhost:6333",
    help="Qdrant host to connect to",
)


class BatchEmbedder:
    def __init__(self):
        self.openai = OpenAI()

    def __call__(
        self, batch: dict[str, np.ndarray], embedding_model: str
    ) -> dict[str, np.ndarray]:
        response = self.openai.embeddings.create(
            input=batch["task"], model=embedding_model
        )
        batch["task_embedding"] = np.array([item.embedding for item in response.data])
        return batch


def main(argv):
    del argv  # Unused.

    # Load examples
    ds = ray.data.read_json(INPUT_FILE)
    logging.info(f"Loaded {ds.count()} examples from {INPUT_FILE}")

    # Create embeddings
    ds = ds.map_batches(
        BatchEmbedder,
        batch_size=20,
        concurrency=1,
        fn_kwargs={"embedding_model": FLAGS.embedding_model},
    )

    # Index embeddings
    qdrant = QdrantClient(FLAGS.qdrant_host)

    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=rest.VectorParams(
            distance=rest.Distance.COSINE,
            size=FLAGS.embedding_dim,
        ),
    )
    logging.info(f"Recreated collection {COLLECTION_NAME}")

    # Record oriented upload
    qdrant.upload_points(
        collection_name=COLLECTION_NAME,
        points=[
            rest.PointStruct(
                id=i,
                vector=row["task_embedding"],
                payload={k: v for k, v in row.items() if k != "task_embedding"},
            )
            for i, row in enumerate(ds.iter_rows())
        ],
    )
    logging.info(f"Uploaded {ds.count()} points to collection {COLLECTION_NAME}")


if __name__ == "__main__":
    app.run(main)
