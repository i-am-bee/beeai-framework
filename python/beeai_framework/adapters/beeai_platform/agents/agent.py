# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0
from typing import Unpack

try:
    from beeai_framework.adapters.acp.agents.agent import ACPAgent
    from beeai_framework.adapters.acp.agents.events import (
        ACPAgentErrorEvent,
        ACPAgentUpdateEvent,
    )
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [beeai-platform] not found.\nRun 'pip install \"beeai-framework[beeai-platform]\"' to install."
    ) from e

from beeai_framework.adapters.beeai_platform.agents.events import (
    BeeAIPlatformAgentErrorEvent,
    BeeAIPlatformAgentUpdateEvent,
    beeai_platform_agent_event_types,
)
from beeai_framework.adapters.beeai_platform.agents.types import (
    BeeAIPlatformAgentRunOutput,
)
from beeai_framework.agents import AgentError, AgentOptions, BaseAgent
from beeai_framework.backend.message import AnyMessage
from beeai_framework.context import RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.emitter.emitter import EventMeta
from beeai_framework.memory import BaseMemory
from beeai_framework.runnable import runnable_entry
from beeai_framework.utils import AbortSignal
from beeai_framework.utils.strings import to_safe_word


class BeeAIPlatformAgent(BaseAgent[BeeAIPlatformAgentRunOutput]):
    def __init__(self, agent_name: str, *, url: str, memory: BaseMemory) -> None:
        super().__init__()
        self._agent = ACPAgent(agent_name=agent_name, url=url, memory=memory)

    @runnable_entry
    async def run(
        self, input: str | AnyMessage | list[str] | list[AnyMessage], /, **kwargs: Unpack[AgentOptions]
    ) -> BeeAIPlatformAgentRunOutput:
        async def handler(context: RunContext) -> BeeAIPlatformAgentRunOutput:
            async def update_event(data: ACPAgentUpdateEvent, event: EventMeta) -> None:
                await context.emitter.emit(
                    "update",
                    BeeAIPlatformAgentUpdateEvent(key=data.key, value=data.value),
                )

            async def error_event(data: ACPAgentErrorEvent, event: EventMeta) -> None:
                await context.emitter.emit(
                    "error",
                    BeeAIPlatformAgentErrorEvent(message=data.message),
                )

            response = await (
                self._agent.run(input, signal=kwargs.get("signal", AbortSignal()))
                .on("update", update_event)
                .on("error", error_event)
            )

            return BeeAIPlatformAgentRunOutput(output=response.output, event=response.event)

        return await handler(RunContext.get())

    async def check_agent_exists(
        self,
    ) -> None:
        try:
            await self._agent.check_agent_exists()
        except Exception as e:
            raise AgentError("Can't connect to beeai platform agent.", cause=e)

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["beeai_platform", "agent", to_safe_word(self._agent._name)],
            creator=self,
            events=beeai_platform_agent_event_types,
        )

    @property
    def memory(self) -> BaseMemory:
        return self._agent.memory

    @memory.setter
    def memory(self, memory: BaseMemory) -> None:
        self._agent.memory = memory

    async def clone(self) -> "BeeAIPlatformAgent":
        cloned = BeeAIPlatformAgent(self._agent._name, url=self._agent._url, memory=await self._agent.memory.clone())
        cloned.emitter = await self.emitter.clone()
        return cloned
