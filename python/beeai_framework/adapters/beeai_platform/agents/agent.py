# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing import Unpack

import httpx

try:
    import a2a.types as a2a_types

    from beeai_framework.adapters.a2a.agents import A2AAgent, A2AAgentErrorEvent, A2AAgentUpdateEvent
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
    BeeAIPlatformAgentOutput,
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


class BeeAIPlatformAgent(BaseAgent[BeeAIPlatformAgentOutput]):
    def __init__(
        self, *, url: str | None = None, agent_card: a2a_types.AgentCard | None = None, memory: BaseMemory
    ) -> None:
        super().__init__()
        self._agent = A2AAgent(url=url, agent_card=agent_card, memory=memory)

    @property
    def name(self) -> str:
        return self._agent.name

    @runnable_entry
    async def run(
        self, input: str | AnyMessage | list[AnyMessage] | a2a_types.Message, /, **kwargs: Unpack[AgentOptions]
    ) -> BeeAIPlatformAgentOutput:
        async def handler(context: RunContext) -> BeeAIPlatformAgentOutput:
            async def update_event(data: A2AAgentUpdateEvent, event: EventMeta) -> None:
                await context.emitter.emit(
                    "update",
                    BeeAIPlatformAgentUpdateEvent(key="update", value=data.value),
                )

            async def error_event(data: A2AAgentErrorEvent, event: EventMeta) -> None:
                await context.emitter.emit(
                    "error",
                    BeeAIPlatformAgentErrorEvent(message=data.message),
                )

            response = await (
                self._agent.run(input, signal=kwargs.get("signal", AbortSignal()))
                .on("update", update_event)
                .on("error", error_event)
            )

            return BeeAIPlatformAgentOutput(output=response.output, event=response.event)

        return await handler(RunContext.get())

    async def check_agent_exists(
        self,
    ) -> None:
        try:
            await self._agent.check_agent_exists()
        except Exception as e:
            raise AgentError("Can't connect to beeai platform agent.", cause=e)

    @classmethod
    async def from_platform(cls, url: str, memory: BaseMemory) -> list["BeeAIPlatformAgent"]:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{url}/api/v1/providers")

            response.raise_for_status()
            return [
                BeeAIPlatformAgent(
                    agent_card=a2a_types.AgentCard(**provider["agent_card"]), memory=await memory.clone()
                )
                for provider in response.json().get("items", [])
            ]

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
        cloned = BeeAIPlatformAgent(url=self._agent._url, memory=await self._agent.memory.clone())
        cloned.emitter = await self.emitter.clone()
        return cloned
