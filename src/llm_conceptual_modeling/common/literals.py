import ast
from typing import Any


def parse_python_literal(value: Any) -> Any:
    if isinstance(value, str):
        return ast.literal_eval(value)
    return value
