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


from beeai_framework.backend.message import AnyMessage
from beeai_framework.memory.base_memory import BaseMemory


class ReadOnlyMemory(BaseMemory):
    """Read-only wrapper for a memory instance."""

    def __init__(self, source: BaseMemory) -> None:
        self._source = source

    @property
    def messages(self) -> list[AnyMessage]:
        return self._source.messages

    async def add(self, message: AnyMessage, index: int | None = None) -> None:
        pass  # No-op for read-only memory

    async def delete(self, message: AnyMessage) -> bool:
        return False  # No-op for read-only memory

    def reset(self) -> None:
        pass  # No-op for read-only memory

    def as_read_only(self) -> "ReadOnlyMemory":
        """Return self since already read-only."""
        return self

    async def clone(self) -> "ReadOnlyMemory":
        return ReadOnlyMemory(await self._source.clone())
