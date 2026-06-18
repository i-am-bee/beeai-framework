# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import AsyncGenerator
from typing import Any

from beeai_framework.backend import (
    AnyMessage,
    AssistantMessage,
    ChatModel,
    ChatModelOutput,
    MessageToolCallContent,
)
from beeai_framework.backend.types import ChatModelInput
from beeai_framework.context import RunContext
from beeai_framework.tools import tool


class ScriptedChatModel(ChatModel):
    """A ChatModel that replays a pre-scripted sequence of outputs (no real LLM).

    Each ``run`` consumes the next list of messages from ``responses``. When the script is
    exhausted it raises (to surface test mistakes) unless ``repeat_last`` is set, in which
    case the final response is replayed indefinitely.
    """

    model_id = "scripted_model"
    # pyrefly: ignore [bad-override]
    provider_id = "ollama"

    def __init__(self, responses: list[list[AnyMessage]], *, repeat_last: bool = False) -> None:
        super().__init__()
        self._responses = list(responses)
        self._repeat_last = repeat_last
        self._last: list[AnyMessage] = []
        self.inputs: list[ChatModelInput] = []

    @property
    def call_count(self) -> int:
        return len(self.inputs)

    # pyrefly: ignore [bad-param-name-override]
    async def _create(self, input: ChatModelInput, _: RunContext) -> ChatModelOutput:
        self.inputs.append(input)
        if self._responses:
            self._last = self._responses.pop(0)
        elif not self._repeat_last:
            raise AssertionError("ScriptedChatModel ran out of scripted responses")
        return ChatModelOutput(output=list(self._last))

    # pyrefly: ignore [bad-param-name-override]
    async def _create_stream(self, input: ChatModelInput, context: RunContext) -> AsyncGenerator[ChatModelOutput]:
        yield await self._create(input, context)


def tool_call_message(tool_name: str, args: dict[str, Any], *, call_id: str = "call_1") -> AssistantMessage:
    """Build an assistant message that requests a single tool call."""
    return AssistantMessage(MessageToolCallContent(id=call_id, tool_name=tool_name, args=json.dumps(args)))


def final_answer_message(text: str) -> AssistantMessage:
    """Build an assistant message that calls the ToolCallingAgent ``final_answer`` tool."""
    return tool_call_message("final_answer", {"response": text}, call_id="call_final")


@tool()
def weather_tool(city: str) -> str:
    """Returns the weather for a city."""

    return f"sunny in {city}"
