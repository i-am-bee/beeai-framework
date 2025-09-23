# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0
import os

import pytest
from pydantic import BaseModel

from beeai_framework.backend import (
    AssistantMessage,
    ChatModelParameters,
    CustomMessage,
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
        models = ["transformers:google/gemma-3-4b-it", "transformers:ibm-granite/granite-3.3-2b-instruct"]
        self.chat_model = ChatModel.from_name(models[-1], ChatModelParameters(temperature=0))

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_local_llm_chat_model_create_user_message(self) -> None:
        response = await self.chat_model.create(
            messages=[UserMessage("How many islands make up the country of Cape Verde?")],
            stream=False,
        )
        assert len(response.messages) == 1
        assert len(response.messages[0].content) > 0
        assert all(isinstance(message, AssistantMessage) for message in response.messages)
        assert "10" in response.messages[0].text

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_local_llm_chat_model_create_stream(self) -> None:
        response = await self.chat_model.create(
            messages=[UserMessage("How many islands make up the country of Cape Verde?")],
            stream=True,
        )

        assert len(response.messages) == 1
        assert len(response.messages[0].content) > 0
        assert all(isinstance(message, AssistantMessage) for message in response.messages)
        assert "10" in response.messages[0].text

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_local_llm_chat_model_create_structure(self) -> None:
        user_message = UserMessage("tell me something interesting")
        custom_message = CustomMessage(role="custom", content="this is a custom message")

        class ReverseWordsSchema(BaseModel):
            reversed: str

        response = await self.chat_model.create_structure(
            schema=ReverseWordsSchema,
            messages=[user_message, custom_message],
        )

        assert isinstance(response.object, dict)
