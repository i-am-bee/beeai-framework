# Copyright 2025 IBM Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from collections.abc import Iterable

from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import AnyMessage, SystemMessage, UserMessage
from beeai_framework.memory.base_memory import BaseMemory


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

Summary:""".format("\n".join([f"{msg.role}: {msg.text}" for msg in messages]))
        )

        response = await self._model.create(messages=[prompt])

        return response.messages[0].get_texts()[0].text

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
