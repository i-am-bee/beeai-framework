# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncGenerator

import pytest

from beeai_framework.backend import (
    AssistantMessage,
    ChatModel,
    ChatModelOutput,
    SystemMessage,
    UserMessage,
)
from beeai_framework.backend.constants import ProviderName
from beeai_framework.backend.types import ChatModelInput
from beeai_framework.context import RunContext
from beeai_framework.memory.summarize_memory import SummarizeMemory


class _StubSummaryModel(ChatModel):
    """Minimal ChatModel that returns a fixed summary and records the prompts it received."""

    def __init__(self, summary: str = "SUMMARY") -> None:
        super().__init__()
        self._summary = summary
        self.prompts: list[str] = []

    @property
    def model_id(self) -> str:
        return "stub-summary"

    @property
    def provider_id(self) -> ProviderName:
        return "ollama"

    async def _create(self, input: ChatModelInput, run: RunContext) -> ChatModelOutput:
        self.prompts.append(input.messages[-1].text)
        return ChatModelOutput(output=[AssistantMessage(self._summary)])

    async def _create_stream(self, input: ChatModelInput, run: RunContext) -> AsyncGenerator[ChatModelOutput]:
        yield await self._create(input, run)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_add_replaces_messages_with_summary() -> None:
    memory = SummarizeMemory(_StubSummaryModel("the summary"))

    await memory.add(UserMessage("hello"))

    assert len(memory.messages) == 1
    assert isinstance(memory.messages[0], SystemMessage)
    assert memory.messages[0].text == "the summary"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_summary_prompt_includes_prior_messages() -> None:
    model = _StubSummaryModel()
    memory = SummarizeMemory(model)

    await memory.add(UserMessage("first message"))
    await memory.add(UserMessage("second message"))

    # The second summarization must see both the prior summary and the new message.
    last_prompt = model.prompts[-1]
    assert "SUMMARY" in last_prompt
    assert "second message" in last_prompt


@pytest.mark.asyncio
@pytest.mark.unit
async def test_add_many_summarizes_into_single_message() -> None:
    model = _StubSummaryModel()
    memory = SummarizeMemory(model)

    await memory.add_many([UserMessage("alpha"), UserMessage("beta")])

    assert len(memory.messages) == 1
    assert isinstance(memory.messages[0], SystemMessage)
    assert "alpha" in model.prompts[-1]
    assert "beta" in model.prompts[-1]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_delete_and_reset() -> None:
    memory = SummarizeMemory(_StubSummaryModel())
    await memory.add(UserMessage("hello"))

    summary_message = memory.messages[0]
    assert await memory.delete(summary_message) is True
    assert await memory.delete(summary_message) is False

    await memory.add(UserMessage("again"))
    memory.reset()
    assert memory.messages == []


@pytest.mark.asyncio
@pytest.mark.unit
async def test_clone_is_independent() -> None:
    memory = SummarizeMemory(_StubSummaryModel())
    await memory.add(UserMessage("hello"))

    clone = await memory.clone()
    clone.reset()

    assert len(memory.messages) == 1
    assert clone.messages == []
