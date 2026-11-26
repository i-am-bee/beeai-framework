# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any, Literal

import pytest
from pydantic import BaseModel, ValidationError
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


@tool()
def tool_literal(kind: Literal["fixed"]) -> str:
    """Tool with a literal-valued parameter."""

    return kind


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


@pytest.mark.unit
@pytest.mark.parametrize("strict", [True, False])
@pytest.mark.parametrize("tools", [[tool_sum], [tool_sum, tool_greet]])
@pytest.mark.parametrize(("wrapped", "parallel"), [(False, False), (True, False), (False, True), (True, True)])
def test_guided_tool_names(tools: list[AnyTool], strict: bool, wrapped: bool, parallel: bool) -> None:
    result, validator = generate_tool_union_schema(
        tools,
        strict=strict,
        allow_parallel_tool_calls=parallel,
        allow_top_level_union=not wrapped,
        use_enum_for_tool_names=True,
    )
    schema = result["json_schema"]["schema"]
    multiple = len(tools) > 1
    if multiple and wrapped:
        schema = schema["properties"]["item"]
    if multiple and parallel:
        schema = schema["items"]
    variants = schema.get("anyOf", [schema])
    for variant, expected in zip(variants, tools, strict=True):
        name = variant["properties"]["name"]
        assert name["enum"] == [expected.name]
        assert "const" not in name

    def validate(name: str, arguments: dict[str, int]) -> None:
        call = {"name": name, "parameters": arguments}
        value = [call] if multiple and parallel else call
        validator.model_validate({"item": value} if multiple and wrapped else value)

    validate("tool_sum", {"a": 1, "b": 2})
    with pytest.raises(ValidationError):
        validate("final_answer", {"a": 1, "b": 2})
    if multiple:
        validate("tool_greet", {"name": 1})
        with pytest.raises(ValidationError):
            validate("tool_sum", {"name": 1})


@pytest.mark.unit
def test_guided_tool_parameter_literal_unchanged() -> None:
    def parameters(use_enum: bool) -> dict:
        result, _ = generate_tool_union_schema(
            [tool_literal],
            strict=False,
            allow_parallel_tool_calls=False,
            allow_top_level_union=True,
            use_enum_for_tool_names=use_enum,
        )
        return result["json_schema"]["schema"]["properties"]["parameters"]

    assert parameters(True) == parameters(False)
    assert parameters(True)["properties"]["kind"]["const"] == "fixed"


@pytest.mark.unit
def test_empty_guided_tool_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tool_sum, "name", "")
    with pytest.raises(ValueError, match="must not be empty"):
        generate_tool_union_schema(
            [tool_sum],
            strict=False,
            allow_parallel_tool_calls=False,
            allow_top_level_union=True,
            use_enum_for_tool_names=True,
        )
