# Copyright 2025 © BeeAI a Series of LF Projects, LLC
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

from collections.abc import Callable
from typing import Any, Self

from acp_sdk import Message, MessageAwaitRequest, MessagePart
from acp_sdk.server import Context

from beeai_framework.utils.io import setup_io_context


class ACPIOContext:
    def __init__(self, context: Context) -> None:
        self.context = context
        self._cleanup: Callable[[], None] = lambda: None

    def __enter__(self) -> Self:
        self._cleanup = setup_io_context(read=self._read)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._cleanup()
        self._cleanup = lambda: None

    async def _read(self, prompt: str) -> str:
        message = Message(parts=[MessagePart(content=prompt)])
        response = await self.context.yield_async(MessageAwaitRequest(message=message))
        # TODO: handle non-text responses
        return str(response.message) if response else ""
