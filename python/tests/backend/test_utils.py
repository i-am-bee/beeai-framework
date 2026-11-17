# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any

import pytest
from pydantic import BaseModel
from syrupy.assertion import SnapshotAssertion

from beeai_framework.backend.utils import generate_tool_union_schema, inline_schema_refs
from beeai_framework.tools import AnyTool, tool


@tool()
def tool_sum(a: int, b: int = 0) -> int:
    """Sum tool"""

    return a + b


@tool()
def tool_greet(name: int) -> str:
    """Greet tool"""

    return f"Hello {name}!"


@pytest.mark.unit
@pytest.mark.parametrize("strict", [True, False])
def test_generate_tool_union_schema(strict: bool, snapshot: SnapshotAssertion) -> None:
    tools: list[AnyTool] = [tool_sum, tool_greet]
    result, _ = generate_tool_union_schema(
        tools, strict=strict, allow_parallel_tool_calls=False, allow_top_level_union=True
    )

    assert result["json_schema"] == snapshot()


@pytest.mark.unit
@pytest.mark.parametrize("strict", [True, False])
def test_generate_tool_union_schema_non_strict(strict: bool, snapshot: SnapshotAssertion) -> None:
    result, _ = generate_tool_union_schema(
        [tool_sum], strict=strict, allow_parallel_tool_calls=False, allow_top_level_union=True
    )
    assert result["json_schema"] == snapshot()


@pytest.mark.unit
def test_generate_tool_union_wrapped_schema(snapshot: SnapshotAssertion) -> None:
    tools: list[AnyTool] = [tool_sum, tool_greet]
    result, _ = generate_tool_union_schema(
        tools, strict=False, allow_parallel_tool_calls=True, allow_top_level_union=False
    )
    assert result["json_schema"] == snapshot()


class TreeNode(BaseModel):
    name: str
    children: list["TreeNode"] = []


@pytest.mark.unit
def test_inline_schema_refs_non_recursive() -> None:
    schema = {
        "type": "object",
        "properties": {"point": {"$ref": "#/$defs/Point"}},
        "$defs": {"Point": {"type": "object", "properties": {"x": {"type": "number"}}}},
    }

    assert inline_schema_refs(schema) == {
        "type": "object",
        "properties": {"point": {"type": "object", "properties": {"x": {"type": "number"}}}},
    }


@pytest.mark.unit
def test_inline_schema_refs_recursive_model() -> None:
    result = inline_schema_refs(TreeNode.model_json_schema())

    assert type(result) is dict
    assert json.loads(json.dumps(result)) == {
        "type": "object",
        "title": "TreeNode",
        "required": ["name"],
        "properties": {
            "name": {"title": "Name", "type": "string"},
            "children": {"default": [], "items": {"$ref": "#/$defs/TreeNode"}, "title": "Children", "type": "array"},
        },
        "$defs": {"TreeNode": TreeNode.model_json_schema()["$defs"]["TreeNode"]},
    }


@pytest.mark.unit
def test_inline_schema_refs_keeps_only_recursive_definitions() -> None:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {"expr": {"$ref": "#/$defs/Expr"}},
        "required": ["expr"],
        "$defs": {
            "Expr": {"anyOf": [{"$ref": "#/$defs/Literal"}, {"$ref": "#/$defs/Op"}]},
            "Op": {
                "type": "object",
                "properties": {"left": {"$ref": "#/$defs/Expr"}, "right": {"$ref": "#/$defs/Expr"}},
            },
            "Literal": {"type": "object", "properties": {"value": {"type": "number"}}},
        },
    }

    result = inline_schema_refs(schema)

    literal = {"type": "object", "properties": {"value": {"type": "number"}}}
    assert set(result["$defs"]) == {"Expr", "Op"}
    assert result["properties"]["expr"] == {"anyOf": [literal, {"$ref": "#/$defs/Op"}]}
    assert result["$defs"]["Expr"] == {"anyOf": [literal, {"$ref": "#/$defs/Op"}]}
    assert result["$defs"]["Op"]["properties"]["left"] == {"$ref": "#/$defs/Expr"}
    json.dumps(result)
