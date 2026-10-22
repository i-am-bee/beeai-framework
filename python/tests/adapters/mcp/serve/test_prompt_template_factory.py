# Copyright 2026 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import pytest

pytest.importorskip("mcp", reason="Optional module [mcp] not installed.")
from mcp.types import TextContent
from pydantic import BaseModel, Field

from beeai_framework.adapters.mcp.serve.server import _prompt_template_factory
from beeai_framework.template import PromptTemplate


class PromptInput(BaseModel):
    topic: str
    tone: str | None = None
    language: str = "en"
    tags: list[str] = Field(default_factory=list)


def make_template() -> PromptTemplate[PromptInput]:
    return PromptTemplate(schema=PromptInput, template="Write about {{topic}} in {{language}}.")


class TestPromptTemplateFactory:
    @pytest.mark.unit
    def test_required_arguments_match_the_schema(self) -> None:
        prompt = _prompt_template_factory(make_template())

        assert prompt.arguments is not None
        assert {argument.name: argument.required for argument in prompt.arguments} == {
            "topic": True,
            "tone": False,
            "language": False,
            "tags": False,
        }

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_renders_with_only_the_required_arguments(self) -> None:
        prompt = _prompt_template_factory(make_template())

        messages = await prompt.render({"topic": "rust"})

        assert len(messages) == 1
        assert isinstance(messages[0].content, TextContent)
        assert messages[0].content.text == "Write about rust in en."
