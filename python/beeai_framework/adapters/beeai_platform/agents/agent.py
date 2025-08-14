# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0
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
    BeeAIPlatformAgentRunOutput,
)
from beeai_framework.agents.base import BaseAgent
from beeai_framework.agents.errors import AgentError
from beeai_framework.backend.message import (
    AnyMessage,
)
from beeai_framework.context import Run, RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.emitter.emitter import EventMeta
from beeai_framework.memory import BaseMemory
from beeai_framework.utils import AbortSignal
from beeai_framework.utils.strings import to_safe_word


class BeeAIPlatformAgent(BaseAgent[BeeAIPlatformAgentRunOutput]):
    def __init__(
        self, *, url: str | None = None, agent_card: a2a_types.AgentCard | None = None, memory: BaseMemory
    ) -> None:
        super().__init__()
        self._agent = A2AAgent(url=url, agent_card=agent_card, memory=memory)

    @property
    def name(self) -> str:
        return self._agent.name

    def run(
        self,
        input: str | AnyMessage | a2a_types.Message,
        *,
        signal: AbortSignal | None = None,
    ) -> Run[BeeAIPlatformAgentRunOutput]:
        async def handler(context: RunContext) -> BeeAIPlatformAgentRunOutput:
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
                self._agent.run(input=input, signal=signal).on("update", update_event).on("error", error_event)
            )

            return BeeAIPlatformAgentRunOutput(result=response.result, event=response.event)

        return self._to_run(
            handler,
            signal=signal,
            run_params={
                "prompt": input,
                "signal": signal,
            },
        )

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
