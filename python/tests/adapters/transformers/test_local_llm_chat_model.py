# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0
import os

import pytest

from beeai_framework.backend import (
    AssistantMessage,
    ChatModelParameters,
    UserMessage,
)
from beeai_framework.backend.chat import ChatModel


@pytest.mark.skipif(
    not os.getenv("TRANSFORMERS_CHAT_MODEL"),
    reason="The model for Transformers was not set.",
)
class TestTransformersChatModel:
    chat_model: ChatModel

    def setup_method(self) -> None:
        model_name = "transformers:" + os.getenv("TRANSFORMERS_CHAT_MODEL")
        self.chat_model = ChatModel.from_name(model_name, ChatModelParameters(temperature=0))

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_local_llm_chat_model_create_user_message(self) -> None:
        response = await self.chat_model.run(
            [UserMessage("How many islands make up the country of Cape Verde?")],
            tools=None,
            tool_choice=None,
            stream=False,
            max_tokens=1000,
            temperature=0.7,
        )
        assert len(response.output) == 1
        assert len(response.output[0].content) > 0
        assert all(isinstance(message, AssistantMessage) for message in response.output)
        assert "10" in response.output[0].text

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_local_llm_chat_model_create_stream(self) -> None:
        response = await self.chat_model.run(
            [UserMessage("How many islands make up the country of Cape Verde?")], stream=True
        )

        assert len(response.output) == 1
        assert len(response.output[0].content) > 0
        assert all(isinstance(message, AssistantMessage) for message in response.output)
        assert "10" in response.output[0].text
