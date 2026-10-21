# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from pydantic import BaseModel, ValidationError

from beeai_framework.template import (
    PromptTemplate,
    PromptTemplateError,
    PromptTemplateInput,
)

"""
Utility functions and classes
"""


@pytest.fixture
def template() -> PromptTemplate[Any]:
    class TestPromptInputSchema(BaseModel):
        task: str
        count: int

    template = PromptTemplate(
        schema=TestPromptInputSchema,
        template="""This is the task: {{task}}{{count}}""",
    )

    return template


"""
Unit Tests
"""


@pytest.mark.unit
def test_render_valid(template: PromptTemplate[Any]) -> None:
    assert template.render({"task": "Test", "count": 1}) == "This is the task: Test1"


@pytest.mark.unit
def test_render_invalid_missing(template: PromptTemplate[Any]) -> None:
    with pytest.raises(ValidationError):
        template.render({"task": "Test"})


@pytest.mark.unit
def test_render_invalid_type(template: PromptTemplate[Any]) -> None:
    with pytest.raises(ValidationError):
        template.render({"task": 1, "count": 1})


@pytest.mark.unit
def test_render_function(template: PromptTemplate[Any]) -> None:
    class TestPromptInputSchema(BaseModel):
        task: str

    template = PromptTemplate(
        schema=TestPromptInputSchema,
        functions={
            "formatDate": lambda data: datetime.now(ZoneInfo("US/Eastern")).strftime("%A, %B %d, %Y at %I:%M:%S %p")
        },
        template="""{{task}} {{formatDate}}""",
    )

    template.render(TestPromptInputSchema(task="Here is a task!"))


@pytest.mark.unit
def test_render_function_clash(template: PromptTemplate[Any]) -> None:
    class TestPromptInputSchema(BaseModel):
        task: str

    template = PromptTemplate(
        schema=TestPromptInputSchema,
        functions={"task": lambda d: "Clashing task!"},
        template="""{{task}}""",
    )

    with pytest.raises(PromptTemplateError):
        template.render(TestPromptInputSchema(task="Here is a task!"))


@pytest.mark.unit
def test_fork_does_not_mutate_original(template: PromptTemplate[Any]) -> None:
    def customizer(config: PromptTemplateInput[Any]) -> PromptTemplateInput[Any]:
        # A shallow copy shares `defaults` with the config it was made from
        new_config = config.model_copy()
        new_config.template = "Forked: {{task}}{{count}}"
        new_config.defaults["count"] = 2
        return new_config

    forked = template.fork(customizer)

    assert forked.render(task="Test") == "Forked: Test2"
    assert template.render(task="Test", count=1) == "This is the task: Test1"
    assert template.render(task="Test") == "This is the task: Test"


@pytest.mark.unit
def test_fork_in_place_customizer(template: PromptTemplate[Any]) -> None:
    def customizer(config: PromptTemplateInput[Any]) -> None:
        config.template = "Forked: {{task}}{{count}}"

    forked = template.fork(customizer)

    assert forked.render(task="Test", count=1) == "Forked: Test1"
    assert template.render(task="Test", count=1) == "This is the task: Test1"
