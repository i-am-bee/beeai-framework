# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest
from pydantic import BaseModel, Field, ValidationError

from beeai_framework.context import RunContext
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


class DecoratorRecord(BaseModel):
    label: str


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_tool_preserves_nested_model_argument(asynchronous: bool) -> None:
    def render(record: DecoratorRecord) -> str:
        """Return the record label."""
        return record.label

    async def render_async(record: DecoratorRecord) -> str:
        """Return the record label asynchronously."""
        return record.label

    wrapped = tool(render_async if asynchronous else render)
    output = await wrapped.run({"record": {"label": "notebook"}})

    assert output.get_text_content() == "notebook"


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_tool_preserves_models_in_containers(asynchronous: bool) -> None:
    def render(records: list[DecoratorRecord], indexed: dict[str, DecoratorRecord]) -> str:
        """Return labels from a list and a mapping."""
        return f"{records[0].label}:{indexed['selected'].label}"

    async def render_async(records: list[DecoratorRecord], indexed: dict[str, DecoratorRecord]) -> str:
        """Return labels from a list and a mapping asynchronously."""
        return f"{records[0].label}:{indexed['selected'].label}"

    wrapped = tool(render_async if asynchronous else render)
    output = await wrapped.run({"records": [{"label": "notebook"}], "indexed": {"selected": {"label": "pencil"}}})

    assert output.get_text_content() == "notebook:pencil"


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_tool_preserves_custom_schema_field_values_and_aliases(asynchronous: bool) -> None:
    class RecordInput(BaseModel):
        record: DecoratorRecord = Field(alias="document")

    def render(record: DecoratorRecord) -> str:
        """Return the aliased record label."""
        return record.label

    async def render_async(record: DecoratorRecord) -> str:
        """Return the aliased record label asynchronously."""
        return record.label

    wrapped = tool(render_async if asynchronous else render, input_schema=RecordInput)
    output = await wrapped.run({"document": {"label": "notebook"}})

    assert output.get_text_content() == "notebook"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tool_runs_with_keyword_only_default_arguments() -> None:
    @tool
    def render(*, label: str, limit: int = 3) -> str:
        """Return the label and its configured limit."""
        return f"{label}:{limit}"

    output = await render.run({"label": "notebook"})

    assert output.get_text_content() == "notebook:3"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tool_forwards_extra_keyword_arguments() -> None:
    @tool
    def render(label: str, **kwargs: Any) -> str:
        """Return the label and extra keyword arguments."""
        return f"{label}:{kwargs['suffix']}:{kwargs['count']}"

    output = await render.run({"label": "notebook", "suffix": "paper", "count": 2})

    assert output.get_text_content() == "notebook:paper:2"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tool_injects_run_context_with_custom_schema() -> None:
    class LabelInput(BaseModel):
        label: str

    @tool(input_schema=LabelInput, with_context=True)
    def render(label: str, context: RunContext) -> str:
        """Return the label with its tool run context."""
        assert isinstance(context, RunContext)
        return label

    output = await render.run({"label": "notebook"})

    assert output.get_text_content() == "notebook"
