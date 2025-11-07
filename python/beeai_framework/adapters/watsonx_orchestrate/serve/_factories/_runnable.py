# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0
from typing import Any

from beeai_framework.adapters.watsonx_orchestrate.serve.agent import (
    WatsonxOrchestrateServerAgent,
    WatsonxOrchestrateServerAgentEmitFn,
    WatsonxOrchestrateServerAgentMessageEvent,
)
from beeai_framework.backend import AnyMessage, ChatModel
from beeai_framework.runnable import Runnable
from beeai_framework.utils.cloneable import Cloneable


class WatsonxOrchestrateServerRunnable(WatsonxOrchestrateServerAgent[Runnable[Any]]):
    @property
    def model_id(self) -> str:
        return self._agent.model_id if isinstance(self._agent, ChatModel) else "unknown"

    async def _stream(self, input: list[AnyMessage], emit: WatsonxOrchestrateServerAgentEmitFn) -> None:
        cloned_runnable = await self._agent.clone() if isinstance(self._agent, Cloneable) else self._agent
        result = await cloned_runnable.run(input)
        await emit(WatsonxOrchestrateServerAgentMessageEvent(text=result.last_message.text))
