# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import AsyncIterable, Callable
from typing import Any

from beeai_framework.adapters.openai.serve._types import OpenAIEvent
from beeai_framework.backend import AnyMessage, ChatModel
from beeai_framework.runnable import AnyRunnable, RunnableOutput
from beeai_framework.utils.cloneable import Cloneable


class OpenAIModel:
    def __init__(
        self,
        runnable: AnyRunnable,
        *,
        model_id: str,
        stream: Callable[[list[AnyMessage]], AsyncIterable[OpenAIEvent]] | None = None,
        run_options: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self._runnable = runnable
        self._run_options = run_options or {}
        self.model_id = model_id
        self.stream = stream or self._stream

    async def run(self, input: list[AnyMessage]) -> RunnableOutput:
        cloned_runnable = await self._runnable.clone() if isinstance(self._runnable, Cloneable) else self._runnable
        return await cloned_runnable.run(input, **self._run_options)  # type: ignore[no-any-return]

    async def _stream(self, input: list[AnyMessage]) -> AsyncIterable[OpenAIEvent]:
        cloned_runnable = await self._runnable.clone() if isinstance(self._runnable, Cloneable) else self._runnable
        if isinstance(cloned_runnable, ChatModel):
            cloned_runnable.parameters.stream = True

        # pyrefly: ignore [missing-attribute]
        response: RunnableOutput = await cloned_runnable.run(input, **self._run_options)
        yield OpenAIEvent(text=response.last_message.text, finish_reason="stop")
