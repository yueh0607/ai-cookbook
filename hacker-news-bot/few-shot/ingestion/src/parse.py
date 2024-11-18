import os

import instructor
from absl import app, flags
from openai import OpenAI
from openai.types.chat import ChatCompletionUserMessageParam
from pydantic import BaseModel, Field

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "model",
    default="gpt-4o",
    help="OpenAI model to use",
)


class Example(BaseModel):
    task: str = Field(
        ..., description="Description of the task that is to be solved by the SQL query"
    )
    sql: str = Field(..., description="DuckDB SQL query to solve the task")
    explanation: str = Field(
        ...,
        description="Generic explanation of the query syntax found in the surrounding markdown",
    )


class ExampleBank(BaseModel):
    """
    Parse the input markdown string to extract text-to-sql examples with explanations.
    Extract one example per sql code block.
    Be sure to inspect all sql code blocks.
    The generic explanation must be strictly based on the surrounding markdown not your prior knowledge.
    Avoid include example specific details such table name or column name in the explanation.
    """

    examples: list[Example] = Field(..., description="List of examples")


def parse(client: OpenAI, input: str, model: str = "gpt-4o") -> list[Example]:
    return client.chat.completions.create(
        model=model,
        response_model=ExampleBank,
        messages=[ChatCompletionUserMessageParam(content=input, role="user")],
    )


def main(argv):
    del argv  # Unused.

    client = instructor.from_openai(OpenAI())

    with open("data/examples.jsonl", "w") as f:
        for filename in os.listdir("data/query_syntax"):
            with open(os.path.join("data/query_syntax", filename), "r") as f_md:
                markdown = f_md.read()
                example_bank = parse(client=client, input=markdown, model=FLAGS.model)
                for example in example_bank.examples:
                    f.write(example.model_dump_json() + "\n")


if __name__ == "__main__":
    app.run(main)
