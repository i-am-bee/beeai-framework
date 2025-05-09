# Copyright 2025 © BeeAI a Series of LF Projects, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from beeai_framework.backend.utils import generate_tool_union_schema
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
def test_generate_tool_union_schema(strict: bool) -> None:
    tools: list[AnyTool] = [tool_sum, tool_greet]
    result = generate_tool_union_schema(tools, strict=strict)
    assert result["json_schema"] == {
        "name": "ToolCall",
        "schema": {
            "anyOf": [
                {
                    "additionalProperties": False,
                    "properties": {
                        "name": {"const": "tool_sum", "description": "Tool Name", "title": "Name", "type": "string"},
                        "parameters": {
                            "additionalProperties": True,
                            "description": "Tool Parameters",
                            "properties": {
                                "a": {"title": "A", "type": "integer"},
                                "b": {"default": 0, "title": "B", "type": "integer"},
                            },
                            "required": ["a", "b"] if strict else ["a"],
                            "title": "tool_sum",
                            "type": "object",
                        },
                    },
                    "required": ["name", "parameters"],
                    "title": "tool_sum",
                    "type": "object",
                },
                {
                    "additionalProperties": False,
                    "properties": {
                        "name": {"const": "tool_greet", "description": "Tool Name", "title": "Name", "type": "string"},
                        "parameters": {
                            "additionalProperties": True,
                            "description": "Tool Parameters",
                            "properties": {"name": {"title": "Name", "type": "integer"}},
                            "required": ["name"],
                            "title": "tool_greet",
                            "type": "object",
                        },
                    },
                    "required": ["name", "parameters"],
                    "title": "tool_greet",
                    "type": "object",
                },
            ],
            "title": "AvailableTools",
        },
    }


@pytest.mark.unit
@pytest.mark.parametrize("strict", [True, False])
def test_generate_tool_union_schema_non_strict(strict: bool) -> None:
    result = generate_tool_union_schema([tool_sum], strict=strict)
    assert result["json_schema"] == {
        "name": "ToolCall",
        "schema": {
            "additionalProperties": False,
            "properties": {
                "name": {"const": "tool_sum", "description": "Tool Name", "title": "Name", "type": "string"},
                "parameters": {
                    "additionalProperties": True,
                    "description": "Tool Parameters",
                    "properties": {
                        "a": {"title": "A", "type": "integer"},
                        "b": {"default": 0, "title": "B", "type": "integer"},
                    },
                    "required": ["a", "b"] if strict else ["a"],
                    "title": "tool_sum",
                    "type": "object",
                },
            },
            "required": ["name", "parameters"],
            "title": "tool_sum",
            "type": "object",
        },
    }
