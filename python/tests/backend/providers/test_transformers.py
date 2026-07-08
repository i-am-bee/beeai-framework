# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0
import json
import os

import pytest

from beeai_framework.adapters.transformers.backend.chat import TransformersChatModel
from beeai_framework.backend import (
    AnyMessage,
    AssistantMessage,
    MessageToolResultContent,
    ToolMessage,
    UserMessage,
)
from beeai_framework.tools.weather import OpenMeteoTool


@pytest.mark.skipif(
    not os.getenv("TRANSFORMERS_CHAT_MODEL"),
    reason="The model for Transformers was not set.",
)
class TestTransformersChatModel:
    def setup_method(self) -> None:
        self.chat_model = TransformersChatModel(os.environ["TRANSFORMERS_CHAT_MODEL"])
        self.chat_model.parameters.temperature = 0

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_local_llm_chat_model_create_user_message(self) -> None:
        response = await self.chat_model.run(
            [UserMessage("How many islands make up the country of Cape Verde?")],
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

    @pytest.mark.e2e
    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Tool calling is not supported yet.")
    async def test_local_llm_chat_model_tools(self) -> None:
        tool = OpenMeteoTool()
        messages: list[AnyMessage] = [UserMessage("What's the current weather in Boston?")]
        response = await self.chat_model.run(messages, tools=[tool])
        messages.extend(response.output)
        tool_call = response.get_tool_calls()[0]
        assert tool_call.tool_name == tool.name
        tool_result = await tool.run(json.loads(tool_call.args))
        messages.append(
            ToolMessage(
                MessageToolResultContent(
                    tool_name=tool.name, result=tool_result.get_text_content(), tool_call_id=tool_call.id
                )
            )
        )
        response = await self.chat_model.run(messages, tools=[tool])
        assert response.last_message.text != ""
        print(response.last_message.text)
