# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0


import pytest

pytest.importorskip("wikipediaapi", reason="Optional module [wikipedia] not installed.")

from beeai_framework.tools import ToolInputValidationError
from beeai_framework.tools.search.wikipedia import (
    WikipediaTool,
    WikipediaToolInput,
    WikipediaToolOutput,
)

"""
Utility functions and classes
"""


@pytest.fixture
def tool() -> WikipediaTool:
    return WikipediaTool()


"""
E2E Tests
"""


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_call_invalid_input_type(tool: WikipediaTool) -> None:
    with pytest.raises(ToolInputValidationError):
        await tool.run(input={"search": "Bee"})


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_output(tool: WikipediaTool) -> None:
    result = await tool.run(input=WikipediaToolInput(query="bee"))
    assert type(result) is WikipediaToolOutput
    assert "Bees are winged insects closely related to wasps and ants" in result.get_text_content()


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_full_text_output(tool: WikipediaTool) -> None:
    result = await tool.run(input=WikipediaToolInput(query="bee", full_text=True))
    assert type(result) is WikipediaToolOutput
    assert "n-triscosane" in result.get_text_content()


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_section_titles(tool: WikipediaTool) -> None:
    result = await tool.run(input=WikipediaToolInput(query="bee", section_titles=True))
    assert type(result) is WikipediaToolOutput
    assert "Characteristics" in result.get_text_content()


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_alternate_language(tool: WikipediaTool) -> None:
    result = await tool.run(input=WikipediaToolInput(query="bee", language="fr"))
    assert type(result) is WikipediaToolOutput
    assert "Les abeilles (Anthophila) forment un clade d'insectes" in result.get_text_content()
