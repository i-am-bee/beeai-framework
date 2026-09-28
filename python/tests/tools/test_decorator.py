# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

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


@pytest.mark.unit
def test_tool_keyword_only_schema() -> None:
    @tool
    def required_only(*, query: str) -> str:
        """Return the query."""
        return query

    schema = required_only.input_schema
    assert schema.model_validate({"query": "hello"}).query == "hello"
    with pytest.raises(ValidationError):
        schema.model_validate({})
    with pytest.raises(ValidationError):
        schema.model_validate({"query": 1})

    @tool
    def required_and_optional(*, query: str, limit: int = 3) -> str:
        """Return the query."""
        return query

    schema = required_and_optional.input_schema
    assert schema.model_validate({"query": "hello"}).limit == 3
    with pytest.raises(ValidationError):
        schema.model_validate({})
    with pytest.raises(ValidationError):
        schema.model_validate({"query": "hello", "limit": "invalid"})
