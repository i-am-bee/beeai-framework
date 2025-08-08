# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import pytest
from syrupy.assertion import SnapshotAssertion

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
