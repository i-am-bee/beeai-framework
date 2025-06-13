# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0


import pytest

from beeai_framework.tools import StringToolOutput, tool

"""
Unit Tests
"""


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tool_annotation() -> None:
    @tool
    def test_tool(query: str) -> str:
        """
        Search factual and historical information, including biography, history, politics, geography, society, culture,
        science, technology, people, animal species, mathematics, and other subjects.

        Args:
            query: The topic or question to search for on Wikipedia.

        Returns:
            The information found via searching Wikipedia.
        """
        return query

    query = "Hello!"
    result: StringToolOutput = await test_tool.run({"query": query})
    assert result.get_text_content() == query


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tool_annotation_no_params() -> None:
    @tool
    def test_tool() -> str:
        """
        Search factual and historical information, including biography, history, politics, geography, society, culture,
        science, technology, people, animal species, mathematics, and other subjects.
        """
        return "Hello!"

    result: StringToolOutput = await test_tool.run({})
    assert result.get_text_content() == "Hello!"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tool_annotation_empty_desc() -> None:
    @tool
    def test_tool() -> str:
        """"""
        return "Hello!"

    result: StringToolOutput = await test_tool.run({})
    assert result.get_text_content() == "Hello!"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tool_annotation_no_desc() -> None:
    with pytest.raises(ValueError):  # No description provided

        @tool
        def test_tool(query: str) -> str:
            return query

        await test_tool.run({"query": "Hello!"})
