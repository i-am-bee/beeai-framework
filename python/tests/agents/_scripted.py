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
from beeai_framework.backend.constants import ProviderName
from beeai_framework.backend.types import ChatModelInput
from beeai_framework.context import RunContext
from beeai_framework.tools import tool


class ScriptedChatModel(ChatModel):
    """A ChatModel that replays a pre-scripted sequence of outputs (no real LLM).

    Each ``run`` consumes the next list of messages from ``responses``. When the script is
    exhausted it raises (to surface test mistakes) unless ``repeat_last`` is set, in which
    case the final response is replayed indefinitely.
    """

    def __init__(self, responses: list[list[AnyMessage]], *, repeat_last: bool = False) -> None:
        super().__init__()
        self._responses = list(responses)
        self._repeat_last = repeat_last
        self._last: list[AnyMessage] = []
        self.inputs: list[ChatModelInput] = []

    @property
    def model_id(self) -> str:
        return "scripted_model"

    @property
    def provider_id(self) -> ProviderName:
        return "ollama"

    @property
    def call_count(self) -> int:
        return len(self.inputs)

    async def _create(self, input: ChatModelInput, run: RunContext) -> ChatModelOutput:
        self.inputs.append(input)
        if self._responses:
            self._last = self._responses.pop(0)
        elif not self._repeat_last:
            raise AssertionError("ScriptedChatModel ran out of scripted responses")
        return ChatModelOutput(output=list(self._last))

    async def _create_stream(self, input: ChatModelInput, run: RunContext) -> AsyncGenerator[ChatModelOutput]:
        yield await self._create(input, run)

    async def clone(self) -> "ScriptedChatModel":
        # Agents clone their llm when cloned; without this the clone would share our state.
        cloned = ScriptedChatModel([list(response) for response in self._responses], repeat_last=self._repeat_last)
        cloned._last = list(self._last)
        cloned.inputs = list(self.inputs)
        return cloned


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
