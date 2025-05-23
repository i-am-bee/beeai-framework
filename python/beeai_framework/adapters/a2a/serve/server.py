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

from collections.abc import Sequence
from typing import Self

import uvicorn
from pydantic import BaseModel
from typing_extensions import TypedDict, TypeVar, Unpack, override

try:
    import a2a.server as a2a_server
    import a2a.server.agent_execution as a2a_agent_execution
    import a2a.server.apps as a2a_apps
    import a2a.server.events as a2a_server_events
    import a2a.server.request_handlers as a2a_request_handlers
    import a2a.server.tasks as a2a_server_tasks
    import a2a.types as a2a_types
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [a2a] not found.\nRun 'pip install \"beeai-framework[a2a]\"' to install."
    ) from e

from beeai_framework.adapters.a2a.serve.agent_executor import BaseA2AAgentExecutor, TollCallingAgentExecutor
from beeai_framework.agents import AnyAgent
from beeai_framework.agents.tool_calling.agent import ToolCallingAgent
from beeai_framework.serve.server import Server
from beeai_framework.utils import ModelLike
from beeai_framework.utils.models import to_model

AnyAgentLike = TypeVar("AnyAgentLike", bound=AnyAgent, default=AnyAgent)


class A2AServerConfig(BaseModel):
    """Configuration for the A2AServer."""

    host: str = "0.0.0.0"
    port: int = 9999


class A2AServerMetadata(TypedDict, total=False):
    name: str
    description: str
    url: str
    version: str
    defaultInputModes: list[str]
    defaultOutputModes: list[str]
    capabilities: a2a_types.AgentCapabilities
    skills: list[a2a_types.AgentSkill]
    queue_manager: a2a_server_events.QueueManager | None
    push_notifier: a2a_server_tasks.PushNotifier | None
    request_context_builder: a2a_agent_execution.RequestContextBuilder | None


class A2AServer(
    Server[
        AnyAgentLike,
        BaseA2AAgentExecutor,
        A2AServerConfig,
    ],
):
    def __init__(self, *, config: ModelLike[A2AServerConfig] | None = None) -> None:
        super().__init__(config=to_model(A2AServerConfig, config or A2AServerConfig()))
        self._metadata_by_agent: dict[AnyAgentLike, A2AServerMetadata] = {}

    def serve(self) -> None:
        if len(self._members) == 0:
            raise ValueError("No agents registered to the server.")

        member = self._members[0]
        factory = type(self)._factories[type(member)]
        config = self._metadata_by_agent.get(member, {})
        executor = factory(member, metadata=config)  # type: ignore[call-arg]

        request_handler = a2a_request_handlers.DefaultRequestHandler(
            agent_executor=executor,
            task_store=a2a_server.tasks.InMemoryTaskStore(),
            queue_manager=config.get("queue_manager", None),
            push_notifier=config.get("push_notifier", None),
            request_context_builder=config.get("request_context_builder", None),
        )

        server = a2a_apps.A2AStarletteApplication(agent_card=executor.agent_card, http_handler=request_handler)
        uvicorn.run(server.build(), host=self._config.host, port=self._config.port)

    @override
    def register(self, input: AnyAgentLike, **metadata: Unpack[A2AServerMetadata]) -> Self:
        if len(self._members) != 0:
            raise ValueError("A2AServer only supports one agent.")
        else:
            super().register(input)
            self._metadata_by_agent[input] = metadata
            return self

    @override
    def register_many(self, input: Sequence[AnyAgentLike]) -> Self:
        raise NotImplementedError("register_many is not implemented for A2AServer")


def _tool_calling_agent_factory(
    agent: ToolCallingAgent, *, metadata: A2AServerMetadata | None = None
) -> BaseA2AAgentExecutor:
    if metadata is None:
        metadata = {}

    return TollCallingAgentExecutor(
        agent=agent,
        agent_card=a2a_types.AgentCard(
            name=metadata.get("name", agent.meta.name),
            description=metadata.get("description", agent.meta.description),
            url=metadata.get("url", "http://localhost:9999"),
            version=metadata.get("version", "1.0.0"),
            defaultInputModes=metadata.get("defaultInputModes", ["text"]),
            defaultOutputModes=metadata.get("defaultOutputModes", ["text"]),
            capabilities=metadata.get("capabilities", a2a_types.AgentCapabilities(streaming=True)),
            skills=metadata.get(
                "skills",
                [
                    a2a_types.AgentSkill(
                        id=metadata.get("name", agent.meta.name),
                        description=metadata.get("description", agent.meta.description),
                        name=metadata.get("name", agent.meta.name),
                        tags=[],
                    )
                ],
            ),
        ),
    )


A2AServer.register_factory(ToolCallingAgent, _tool_calling_agent_factory)
