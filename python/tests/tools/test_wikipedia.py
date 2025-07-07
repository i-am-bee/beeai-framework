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

"""
E2E Tests
"""


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_call_invalid_input_type() -> None:
    tool = WikipediaTool()
    with pytest.raises(ToolInputValidationError):
        await tool.run(input={"search": "Bee"})


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_output() -> None:
    tool = WikipediaTool()
    result = await tool.run(input=WikipediaToolInput(query="bee"))
    assert type(result) is WikipediaToolOutput
    assert "Bees are winged" in result.get_text_content()


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_full_text_output() -> None:
    tool = WikipediaTool()
    result = await tool.run(input=WikipediaToolInput(query="bee", full_text=True))
    assert type(result) is WikipediaToolOutput
    assert "n-triscosane" in result.get_text_content()


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_alternate_language() -> None:
    tool = WikipediaTool(language="fr")
    result = await tool.run(input=WikipediaToolInput(query="bee"))
    assert isinstance(result, WikipediaToolOutput)
    print(result.get_text_content())
    assert "Les abeilles (Anthophila) forment un clade d'insectes" in result.get_text_content()
