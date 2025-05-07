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

import acp_sdk.models as acp_models
import acp_sdk.server.context as acp_context
import acp_sdk.server.server as acp_server
import acp_sdk.server.types as acp_types
from acp_sdk.server.agent import Agent as AcpBaseAgent
from pydantic import BaseModel

from beeai_framework.adapters.acp.serve.utils import to_framework_message
from beeai_framework.agents.react.agent import ReActAgent
from beeai_framework.agents.react.events import ReActAgentUpdateEvent
from beeai_framework.agents.tool_calling.agent import ToolCallingAgent
from beeai_framework.agents.tool_calling.events import ToolCallingAgentSuccessEvent
from beeai_framework.backend.message import (
    AnyMessage,
    Role,
)
from beeai_framework.serve.server import AgentServer
from beeai_framework.utils.lists import find_index


class AcpAgent(AcpBaseAgent):
    """A wrapper for a BeeAI agent to be used with the ACP server."""

    def __init__(
        self,
        fn: Callable[
            [list[acp_models.Message], acp_context.Context],
            AsyncGenerator[acp_types.RunYield, acp_types.RunYieldResume],
        ],
        name: str,
        description: str | None = None,
        metadata: acp_models.Metadata | None = None,
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
    def metadata(self) -> acp_models.Metadata:
        return self._metadata or acp_models.Metadata()

    async def run(
        self, input: list[acp_models.Message], context: acp_context.Context
    ) -> AsyncGenerator[acp_types.RunYield, acp_types.RunYieldResume]:
        try:
            gen: AsyncGenerator[acp_types.RunYield, acp_types.RunYieldResume] = self.fn(input, context)
            value = None
            while True:
                value = yield await gen.asend(value)
        except StopAsyncIteration:
            pass


class AcpServerConfig(BaseModel):
    """Configuration for the AcpServer."""

    host: str = "127.0.0.1"
    port: int = 8000


class AcpAgentServer(AgentServer[AcpServerConfig, AcpAgent]):
    def __init__(self, *, config: AcpServerConfig | None = None) -> None:
        super().__init__(config=config or AcpServerConfig())
        self._server = acp_server.Server()

    def serve(self) -> None:
        for agent in self.agents:
            factory = type(self)._factories[type(agent)]
            self._server.register(factory(agent))

        self._server.run(
            host=self._config.host,
            port=self._config.port,
        )


def _register_react_agent(agent: ReActAgent) -> AcpAgent:
    async def run(
        input: list[acp_models.Message], context: acp_context.Context
    ) -> AsyncGenerator[acp_types.RunYield, acp_types.RunYieldResume]:
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
                            yield acp_models.MessagePart(content=update, role="assistant")  # type: ignore[call-arg]

    return AcpAgent(fn=run, name=acp_models.AgentName(agent.meta.name), description=agent.meta.description)


AcpAgentServer.register_factory(ReActAgent, _register_react_agent)


def _register_tool_calling_agent(agent: ToolCallingAgent) -> AcpAgent:
    async def run(
        input: list[acp_models.Message], context: acp_context.Context
    ) -> AsyncGenerator[acp_types.RunYield, acp_types.RunYieldResume]:
        framework_messages = [
            to_framework_message(Role(message.parts[0].role), str(message))  # type: ignore[attr-defined]
            for message in input
        ]
        await agent.memory.add_many(framework_messages)

        last_msg: AnyMessage | None = None
        async for data, _ in agent.run():
            messages = data.state.memory.messages
            if last_msg is None:
                last_msg = messages[-1]

            cur_index = find_index(messages, lambda msg: msg is last_msg, fallback=-1, reverse_traversal=True)  # noqa: B023
            for message in messages[cur_index + 1 :]:
                yield {"message": message.to_plain()}
                last_msg = message

            if isinstance(data, ToolCallingAgentSuccessEvent) and data.state.result is not None:
                yield acp_models.MessagePart(content=data.state.result.text, role="assistant")  # type: ignore[call-arg]

    return AcpAgent(fn=run, name=acp_models.AgentName(agent.meta.name), description=agent.meta.description)


AcpAgentServer.register_factory(ToolCallingAgent, _register_tool_calling_agent)
