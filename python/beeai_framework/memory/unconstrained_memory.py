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


from copy import copy
from typing import Any

from beeai_framework.backend.message import AnyMessage
from beeai_framework.memory.base_memory import BaseMemory


class UnconstrainedMemory(BaseMemory):
    """Simple memory implementation with no constraints."""

    def __init__(self) -> None:
        self._messages: list[AnyMessage] = []

    @property
    def messages(self) -> list[AnyMessage]:
        return self._messages

    async def add(self, message: AnyMessage, index: int | None = None) -> None:
        index = len(self._messages) if index is None else max(0, min(index, len(self._messages)))
        self._messages.insert(index, message)

    async def delete(self, message: AnyMessage) -> bool:
        try:
            self._messages.remove(message)
            return True
        except ValueError:
            return False

    def reset(self) -> None:
        self._messages.clear()

    def create_snapshot(self) -> dict[str, Any]:
        return {"messages": copy(self._messages)}

    def load_snapshot(self, state: dict[str, Any]) -> None:
        self._messages = copy(state["messages"])
