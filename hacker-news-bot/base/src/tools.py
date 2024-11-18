import inspect
import logging
from typing import Any, Callable, Optional

import duckdb
from duckdb import DuckDBPyConnection
from langfuse.decorators import observe
from openai.types.chat import ChatCompletionToolParam
from openai.types.shared_params import FunctionDefinition
from pydantic import create_model

con: Optional[DuckDBPyConnection] = None


def get_connection() -> DuckDBPyConnection:
    global con
    if con is None:
        con = duckdb.connect(":memory:")
        con.execute("""
CREATE TABLE IF NOT EXISTS posts AS
SELECT * FROM read_parquet('data/hacker_news.parquet');
""")

    return con


@observe()
def run_sql(query: str) -> str:
    """Run DuckDB SQL query against Hacker news table `posts` and return the result as a JSON string."""

    con = get_connection()
    df = con.sql(query).fetchdf()

    # Count the number of rows in the result
    if len(df) > 100:
        logging.warning(
            f"The result contains {len(df)} rows. Only returning the first 100."
        )
        df = df.head(100)

    return df.to_json(orient="records")


def get_tool_param(func: Callable[..., Any]) -> ChatCompletionToolParam:
    # Get the signature of the function
    sig = inspect.signature(func)

    # Prepare a dictionary to store the fields for the Pydantic model
    model_fields = {}

    # Loop over the function's parameters and extract the type and default value
    for param_name, param in sig.parameters.items():
        # Get the type hint
        param_type = (
            param.annotation if param.annotation != inspect.Parameter.empty else Any
        )

        # Check if the parameter has a default value
        if param.default != inspect.Parameter.empty:
            model_fields[param_name] = (param_type, param.default)
        else:
            model_fields[param_name] = (param_type, ...)

    # Dynamically create a Pydantic model
    model_name = (
        "".join(word.capitalize() for word in func.__name__.split("_")) + "Model"
    )
    model = create_model(model_name, **model_fields)
    schema = model.model_json_schema()

    return ChatCompletionToolParam(
        function=FunctionDefinition(
            name=func.__name__,
            description=(func.__doc__ or "").strip(),
            parameters=schema,
        ),
        type="function",
    )
