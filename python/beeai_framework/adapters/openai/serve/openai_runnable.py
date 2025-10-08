# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import AsyncIterable, Callable

from beeai_framework.adapters.openai.serve._types import OpenAIEvent
from beeai_framework.backend import AnyMessage
from beeai_framework.runnable import AnyRunnable, RunnableOutput
from beeai_framework.utils.cloneable import Cloneable


class OpenAIRunnable:
    def __init__(
        self,
        runnable: AnyRunnable,
        *,
        model_id: str,
        stream: Callable[[list[AnyMessage]], AsyncIterable[OpenAIEvent]] | None = None,
    ) -> None:
        super().__init__()
        self.runnable: AnyRunnable = runnable
        self.model_id: str = model_id
        self.stream: Callable[[list[AnyMessage]], AsyncIterable[OpenAIEvent]] = stream or self._stream

    async def run(self, input: list[AnyMessage]) -> RunnableOutput:
        cloned_runnable = await self.runnable.clone() if isinstance(self.runnable, Cloneable) else self.runnable
        response: RunnableOutput = await cloned_runnable.run(input)
        return response

    async def _stream(self, input: list[AnyMessage]) -> AsyncIterable[OpenAIEvent]:
        cloned_runnable = await self.runnable.clone() if isinstance(self.runnable, Cloneable) else self.runnable
        response: RunnableOutput = await cloned_runnable.run(input)
        yield OpenAIEvent(text=response.last_message.text, finish_reason="stop")
