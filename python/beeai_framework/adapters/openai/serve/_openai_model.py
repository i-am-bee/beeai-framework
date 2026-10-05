# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import AsyncIterable, Callable
from copy import copy
from typing import TypeVar, cast

from beeai_framework.adapters.openai.serve._types import OpenAIEvent
from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.agents.requirement.requirements.ask_permission import AskPermissionRequirement
from beeai_framework.backend import AnyMessage, ChatModel
from beeai_framework.runnable import AnyRunnable, RunnableOutput
from beeai_framework.utils.cloneable import Cloneable

T = TypeVar("T", bound=AnyRunnable)


async def _clone_runnable(runnable: T) -> T:
    cloned = await runnable.clone() if isinstance(runnable, Cloneable) else runnable
    if isinstance(cloned, RequirementAgent):
        # RequirementAgent.clone shares requirements. Remembered approvals must belong
        # to this invocation, never to another caller of the registered agent.
        cloned._requirements = [
            copy(requirement) if isinstance(requirement, AskPermissionRequirement) else requirement
            for requirement in cloned._requirements
        ]
        for requirement in cloned._requirements:
            if isinstance(requirement, AskPermissionRequirement):
                requirement._state = requirement._state.copy()
    return cast(T, cloned)


class OpenAIModel:
    def __init__(
        self,
        runnable: AnyRunnable,
        *,
        model_id: str,
        stream: Callable[[list[AnyMessage]], AsyncIterable[OpenAIEvent]] | None = None,
    ) -> None:
        super().__init__()
        self._runnable = runnable
        self.model_id = model_id
        self.stream = stream or self._stream

    async def run(self, input: list[AnyMessage]) -> RunnableOutput:
        cloned_runnable = await _clone_runnable(self._runnable)
        return await cloned_runnable.run(input)  # type: ignore[no-any-return]

    async def _stream(self, input: list[AnyMessage]) -> AsyncIterable[OpenAIEvent]:
        cloned_runnable = await _clone_runnable(self._runnable)
        if isinstance(cloned_runnable, ChatModel):
            cloned_runnable.parameters.stream = True

        # pyrefly: ignore [missing-attribute]
        response: RunnableOutput = await cloned_runnable.run(input)
        yield OpenAIEvent(text=response.last_message.text, finish_reason="stop")
