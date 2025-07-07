# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

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
