# Copyright 2025 BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncGenerator

import pytest

from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import AssistantMessage, MessageToolCallContent, UserMessage
from beeai_framework.backend.types import ChatModelInput, ChatModelOutput
from beeai_framework.cache import SlidingCache
from beeai_framework.context import RunContext


class ToolCallRetryDummyModel(ChatModel):
    model_id = "tool_call_retry_model"
    # pyrefly: ignore [bad-override]
    provider_id = "ollama"

    def __init__(self) -> None:
        super().__init__(cache=SlidingCache(size=4))
        self.create_calls = 0

    # pyrefly: ignore [bad-param-name-override]
    async def _create(self, input: ChatModelInput, _: RunContext) -> ChatModelOutput:
        self.create_calls += 1
        if self.create_calls == 1:
            return ChatModelOutput(
                output=[AssistantMessage(MessageToolCallContent(id="call_1", tool_name="test_tool", args="not json"))]
            )

        return ChatModelOutput(output=[AssistantMessage("fixed tool call")])

    # pyrefly: ignore [bad-param-name-override]
    async def _create_stream(self, input: ChatModelInput, context: RunContext) -> AsyncGenerator[ChatModelOutput]:
        yield await self._create(input, context)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_chat_model_clears_cached_invalid_tool_call_before_retry() -> None:
    chat_model = ToolCallRetryDummyModel()

    response = await chat_model.run([UserMessage("call the tool")], max_retries=1)

    assert response.get_text_content() == "fixed tool call"
    assert chat_model.create_calls == 2
