# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable

from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import AnyMessage, AssistantMessage, SystemMessage, ToolMessage, UserMessage
from beeai_framework.memory.base_memory import BaseMemory
from beeai_framework.utils.strings import to_json


def _format_message(message: AnyMessage) -> str:
    """Render a message for the summary prompt, including tool calls and tool results.

    ``Message.text`` only joins text parts, so a tool call or tool result (which is how agents
    such as RequirementAgent store tool use and their final answer) would otherwise be blank.
    """
    parts = [message.text] if message.text else []
    if isinstance(message, AssistantMessage):
        parts.extend(f"[called {call.tool_name}({call.args})]" for call in message.get_tool_calls())
    elif isinstance(message, ToolMessage):
        parts.extend(
            f"[{result.tool_name} returned: "
            f"{result.result if isinstance(result.result, str) else to_json(result.result, sort_keys=False)}]"
            for result in message.get_tool_results()
        )
    return f"{message.role}: {' '.join(parts)}"


class SummarizeMemory(BaseMemory):
    """Memory implementation that summarizes conversations."""

    def __init__(self, model: ChatModel) -> None:
        self._messages: list[AnyMessage] = []
        self._model = model

    @property
    def messages(self) -> list[AnyMessage]:
        return self._messages

    async def add(self, message: AnyMessage, index: int | None = None) -> None:
        """Add a message and trigger summarization if needed."""
        messages_to_summarize = [*self._messages, message]
        summary = await self._summarize_messages(messages_to_summarize)

        self._messages = [SystemMessage(summary)]

    async def add_many(self, messages: Iterable[AnyMessage], start: int | None = None) -> None:
        """Add multiple messages and summarize."""
        messages_to_summarize = self._messages + list(messages)
        summary = await self._summarize_messages(messages_to_summarize)

        self._messages = [SystemMessage(summary)]

    async def _summarize_messages(self, messages: list[AnyMessage]) -> str:
        """Summarize a list of messages using the LLM."""
        if not messages:
            return ""

        prompt = UserMessage(
            """Summarize the following conversation. Be concise but include all key information.

Previous messages:
{}

Summary:""".format("\n".join([_format_message(msg) for msg in messages]))
        )

        response = await self._model.run([prompt])

        # A streamed response merges its chunks into one message with a text part per
        # chunk, so read all of them rather than the first.
        return response.get_text_content()

    async def delete(self, message: AnyMessage) -> bool:
        """Delete a message from memory."""
        try:
            self._messages.remove(message)
            return True
        except ValueError:
            return False

    def reset(self) -> None:
        """Clear all messages from memory."""
        self._messages.clear()

    async def clone(self) -> "SummarizeMemory":
        cloned = SummarizeMemory(await self._model.clone())
        cloned._messages = self._messages.copy()
        return cloned
