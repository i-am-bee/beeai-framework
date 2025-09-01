# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import Any, Self

from a2a.types import TextPart
from beeai_sdk.a2a.types import InputRequired
from beeai_sdk.server.context import RunContext

from beeai_framework.utils.io import setup_io_context


class BeeAIPlatformIOContext:
    def __init__(self, context: RunContext) -> None:
        self.context = context
        self._cleanup: Callable[[], None] = lambda: None

    def __enter__(self) -> Self:
        self._cleanup = setup_io_context(read=self._read)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._cleanup()
        self._cleanup = lambda: None

    async def _read(self, prompt: str) -> str:
        response = await self.context.yield_async(InputRequired(text=prompt))
        parts = response.parts if response else []

        for part in parts:
            if isinstance(part.root, TextPart):
                return part.root.text

        return ""
