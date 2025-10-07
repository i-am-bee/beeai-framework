# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import AsyncIterable, Callable
from typing import Any

from sse_starlette import ServerSentEvent

from beeai_framework.backend import AnyMessage
from beeai_framework.runnable import AnyRunnable, RunnableOutput
from beeai_framework.utils.cloneable import Cloneable


class OpenAIRunnable:
    def __init__(
        self,
        runnable: AnyRunnable,
        *,
        model_id: str,
        handler: Callable[[list[AnyMessage]], AsyncIterable[ServerSentEvent]],
    ) -> None:
        super().__init__()
        self.runnable: AnyRunnable = runnable
        self.model_id: str = model_id
        self.handler: Callable[[list[AnyMessage]], AsyncIterable[ServerSentEvent]] = handler

    async def run(self, input: list[AnyMessage]) -> RunnableOutput:
        cloned_runnable = await self.runnable.clone() if isinstance(self.runnable, Cloneable) else self.runnable
        response: RunnableOutput = await cloned_runnable.run(input)
        return response

    # TODO
    async def stream(self, input: list[AnyMessage]) -> AsyncIterable[Any]:
        cloned_runnable = await self.runnable.clone() if isinstance(self.runnable, Cloneable) else self.runnable
        response = await cloned_runnable.run(input)
        yield response.last_message
