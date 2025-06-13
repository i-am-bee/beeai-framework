# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0


import pytest

pytest.importorskip("duckduckgo_search", reason="Optional module [duckduckgo] not installed.")

from beeai_framework.tools import ToolInputValidationError
from beeai_framework.tools.search.duckduckgo import (
    DuckDuckGoSearchTool,
    DuckDuckGoSearchToolInput,
    DuckDuckGoSearchToolOutput,
)

"""
Utility functions and classes
"""


@pytest.fixture
def tool() -> DuckDuckGoSearchTool:
    return DuckDuckGoSearchTool()


"""
Unit Tests
"""


@pytest.mark.unit
@pytest.mark.asyncio
async def test_call_invalid_input_type(tool: DuckDuckGoSearchTool) -> None:
    with pytest.raises(ToolInputValidationError):
        await tool.run(input={"search": "Poland"})


"""
E2E Tests
"""


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_output(tool: DuckDuckGoSearchTool) -> None:
    result = await tool.run(
        input=DuckDuckGoSearchToolInput(query="What is the highest mountain of the Czech Republic?")
    )
    assert type(result) is DuckDuckGoSearchToolOutput
    print(result.get_text_content())
    assert "Sněžka" in result.get_text_content()
