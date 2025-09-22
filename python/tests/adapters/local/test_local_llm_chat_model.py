# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0
import json
import os
import re

import pytest
from pydantic import BaseModel

from beeai_framework.backend import (
    AnyMessage,
    AssistantMessage,
    ChatModelParameters,
    CustomMessage,
    MessageToolResultContent,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from beeai_framework.backend.chat import ChatModel
from beeai_framework.tools import AnyTool, ToolOutput
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool


@pytest.mark.skipif(
    not os.getenv("TRANSFORMERS_CHAT_MODEL"),
    reason="The model for Transformers was not set.",
)
class TestTransformersChatModel:
    chat_model: ChatModel

    def setup_method(self) -> None:
        self.chat_model = ChatModel.from_name("transformers:google/gemma-3-4b-it", ChatModelParameters(temperature=0))

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
    async def test_local_llm_chat_model_create_tool_message(self) -> None:
        tools: list[AnyTool] = [OpenMeteoTool()]
        tool_messages: list[AnyMessage] = [
            SystemMessage("You are a helpful assistant. Use tools to provide a correct answer."),
            UserMessage("What's the fastest marathon time?"),
        ]
        answer = await self.get_answer_from_chat_model_for_tool_calls(tools=tools, tool_messages=tool_messages)

        assert len(answer) > 0

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

    async def get_answer_from_chat_model_for_tool_calls(
        self, tools: list[AnyTool], tool_messages: list[AnyMessage]
    ) -> str:
        max_iteration = 10
        answer = ""
        iteration = 0
        while iteration < max_iteration:
            response = await self.chat_model.create(
                messages=tool_messages,
                tools=tools,
            )

            tool_calls = response.get_tool_calls()
            tool_messages.extend(response.messages)

            tool_results: list[ToolMessage] = []

            for tool_call in tool_calls:
                print(f"-> running '{tool_call.tool_name}' tool with {tool_call.args}")
                tool: AnyTool = next(tool for tool in tools if tool.name == tool_call.tool_name)
                assert tool is not None
                res: ToolOutput = await tool.run(json.loads(tool_call.args))
                result = res.get_text_content()
                print(
                    f"<- got response from '{tool_call.tool_name}'",
                    re.sub(r"\s+", " ", result)[:256] + " (truncated)",
                )
                tool_results.append(
                    ToolMessage(
                        MessageToolResultContent(
                            result=result,
                            tool_name=tool_call.tool_name,
                            tool_call_id=tool_call.id,
                        )
                    )
                )

            tool_messages.extend(tool_results)

            answer = response.get_text_content()

            if answer:
                break

            iteration += 1

        return answer
