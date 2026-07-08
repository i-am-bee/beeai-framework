# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import pytest

from beeai_framework.tools import JSONToolOutput, ToolInputValidationError
from beeai_framework.tools.weather import OpenMeteoTool, OpenMeteoToolInput

"""
Utility functions and classes
"""


@pytest.fixture
def tool() -> OpenMeteoTool:
    return OpenMeteoTool()


"""
E2E Tests
"""


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_call_model(tool: OpenMeteoTool) -> None:
    await tool.run(
        input=OpenMeteoToolInput(
            location_name="Cambridge",
            country="US",
            temperature_unit="fahrenheit",
        )
    )


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_call_dict(tool: OpenMeteoTool) -> None:
    await tool.run(input={"location_name": "White Plains"})


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_call_invalid_missing_field(tool: OpenMeteoTool) -> None:
    with pytest.raises(ToolInputValidationError):
        await tool.run(input={})


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_call_invalid_bad_type(tool: OpenMeteoTool) -> None:
    with pytest.raises(ToolInputValidationError):
        await tool.run(input={"location_name": 1})


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_output(tool: OpenMeteoTool) -> None:
    result = await tool.run(input={"location_name": "White Plains"})
    assert isinstance(result, JSONToolOutput)
    assert "current" in result.get_text_content()
