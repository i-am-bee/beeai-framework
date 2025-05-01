# Copyright 2025 © BeeAI a Series of LF Projects, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import AsyncGenerator, Callable
from typing import Any

from acp_sdk.models.models import AgentName, MessagePart, Metadata
from acp_sdk.models.models import Message as AcpMessage
from acp_sdk.server.agent import Agent as AcpAgent
from acp_sdk.server.context import Context
from acp_sdk.server.server import Server as AcpHttpServer
from acp_sdk.server.types import RunYield, RunYieldResume
from pydantic import BaseModel

from beeai_framework.agents.base import BaseAgent
from beeai_framework.agents.react.agent import ReActAgent
from beeai_framework.agents.react.events import ReActAgentUpdateEvent
from beeai_framework.agents.tool_calling.agent import ToolCallingAgent
from beeai_framework.backend.message import AssistantMessage, Message, Role, UserMessage
from beeai_framework.serve.server import Server


class AcpServerConfig(BaseModel):
    """Configuration for the AcpServer."""

    host: str = "127.0.0.1"
    port: int = 8000


class Agent(AcpAgent):
    """A wrapper for a BeeAI agent to be used with the ACP server."""

    def __init__(
        self,
        fn: Callable[[list[AcpMessage], Context], AsyncGenerator[RunYield, RunYieldResume]],
        name: str,
        description: str | None = None,
        metadata: Metadata | None = None,
    ) -> None:
        super().__init__()
        self.fn = fn
        self._name = name
        self._description = description
        self._metadata = metadata

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description or ""

    @property
    def metadata(self) -> Metadata:
        return self._metadata or Metadata()

    async def run(self, input: list[AcpMessage], context: Context) -> AsyncGenerator[RunYield, RunYieldResume]:
        try:
            gen: AsyncGenerator[RunYield, RunYieldResume] = self.fn(input, context)
            value = None
            while True:
                value = yield await gen.asend(value)
        except StopAsyncIteration:
            pass


class AcpServer(Server[AcpServerConfig]):
    def __init__(self) -> None:
        super().__init__()
        self.server = AcpHttpServer()

    def serve(self, *, config: AcpServerConfig | None = None) -> None:
        if not config:
            config = AcpServerConfig()
        agents = [self.conver_to_acp_agent(agent) for agent in self.agents]
        self.server.register(*agents)
        self.server.run(
            host=config.host,
            port=config.port,
        )

    def conver_to_acp_agent(self, agent: BaseAgent[Any]) -> Agent:
        """Convert a BeeAI agent to an ACP agent."""

        def to_framework_message(role: Role, content: str) -> Message[Any]:
            match role:
                case Role.USER:
                    return UserMessage(content)
                case Role.ASSISTANT:
                    return AssistantMessage(content)
                case _:
                    raise ValueError(f"Unsupported role {role}")

        if isinstance(agent, ReActAgent):

            async def run(input: list[AcpMessage], context: Context) -> AsyncGenerator[RunYield, RunYieldResume]:
                framework_messages = [
                    to_framework_message(Role(message.parts[0].role), str(message))  # type: ignore[attr-defined]
                    for message in input
                ]
                await agent.memory.add_many(framework_messages)

                async for data, event in agent.run():
                    match (data, event.name):
                        case (ReActAgentUpdateEvent(), "partial_update"):
                            update = data.update.value
                            if not isinstance(update, str):
                                update = update.get_text_content()
                            match data.update.key:
                                case "thought" | "tool_name" | "tool_input" | "tool_output":
                                    yield {data.update.key: update}
                                case "final_answer":
                                    yield MessagePart(content=update, role="assistant")  # type: ignore[call-arg]

            return Agent(fn=run, name=AgentName(agent.meta.name), description=agent.meta.description)

        elif isinstance(agent, ToolCallingAgent):

            async def run(input: list[AcpMessage], context: Context) -> AsyncGenerator[RunYield, RunYieldResume]:
                framework_messages = [
                    to_framework_message(Role(message.parts[0].role), str(message))  # type: ignore[attr-defined]
                    for message in input
                ]
                await agent.memory.add_many(framework_messages)

                async for data, event in agent.run():
                    match event.name:
                        case "start":
                            yield {event.name: "starting new iteration"}
                        case "success":
                            message = data.state.memory.messages[-1]
                            yield {message.role: message.content}
                            if data.state.result:
                                yield MessagePart(content=data.state.result.text, role="assistant")  # type: ignore[call-arg]

            return Agent(fn=run, name=AgentName(agent.meta.name), description=agent.meta.description)
        else:
            raise TypeError("Unsupported agent type. Only ReActAgent and ToolCallingAgent are supported.")
