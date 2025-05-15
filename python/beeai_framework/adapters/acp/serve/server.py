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

from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import Any, Generic, Self

import acp_sdk.models as acp_models
import acp_sdk.server.context as acp_context
import acp_sdk.server.server as acp_server
import acp_sdk.server.types as acp_types
from acp_sdk import AnyModel, Author, Capability, Contributor, Dependency, Link
from typing_extensions import TypedDict, TypeVar, Unpack, override

from beeai_framework.adapters.acp.serve._agent import AcpAgent, AcpServerConfig
from beeai_framework.adapters.acp.serve._utils import acp_msg_to_framework_msg
from beeai_framework.agents import AnyAgent
from beeai_framework.agents.react.agent import ReActAgent
from beeai_framework.agents.react.events import ReActAgentUpdateEvent
from beeai_framework.agents.tool_calling.agent import ToolCallingAgent
from beeai_framework.agents.tool_calling.events import ToolCallingAgentSuccessEvent
from beeai_framework.backend.message import (
    AnyMessage,
    Role,
)
from beeai_framework.serve.server import Server
from beeai_framework.utils import ModelLike
from beeai_framework.utils.lists import find_index
from beeai_framework.utils.models import to_model

AnyAgentLike = TypeVar("AnyAgentLike", bound=AnyAgent, default=AnyAgent)


class AcpAgentServerMetadata(TypedDict, total=False):
    name: str
    description: str
    annotations: AnyModel
    documentation: str
    license: str
    programming_language: str
    natural_languages: list[str]
    framework: str
    capabilities: list[Capability]
    domains: list[str]
    tags: list[str]
    created_at: datetime
    updated_at: datetime
    author: Author
    contributors: list[Contributor]
    links: list[Link]
    dependencies: list[Dependency]
    recommended_models: list[str]
    extra: dict[str, Any]


class AcpAgentServer(Generic[AnyAgentLike], Server[AnyAgentLike, AcpAgent, AcpServerConfig]):
    def __init__(self, *, config: ModelLike[AcpServerConfig] | None = None) -> None:
        super().__init__(config=to_model(AcpServerConfig, config or {}))
        self._metadata_by_agent: dict[AnyAgentLike, AcpAgentServerMetadata] = {}
        self._server = acp_server.Server()

    def serve(self) -> None:
        for member in self.members:
            factory = type(self)._factories[type(member)]
            config = self._metadata_by_agent.get(member, None)
            self._server.register(factory(member, metadata=config))  # type: ignore[call-arg]

        self._server.run(**self._config.model_dump(exclude_unset=True))

    @override
    def register(self, input: AnyAgentLike, **metadata: Unpack[AcpAgentServerMetadata]) -> Self:
        super().register(input)
        if not metadata.get("programming_language"):
            metadata["programming_language"] = "Python"
        if not metadata.get("natural_languages"):
            metadata["natural_languages"] = ["English"]
        if not metadata.get("framework"):
            metadata["framework"] = "BeeAI"
        if not metadata.get("created_at"):
            metadata["created_at"] = datetime.now(tz=UTC)
        if not metadata.get("updated_at"):
            metadata["updated_at"] = datetime.now(tz=UTC)

        self._metadata_by_agent[input] = metadata

        return self


def to_acp_agent_metadata(metadata: AcpAgentServerMetadata) -> acp_models.Metadata:
    copy = metadata.copy()
    copy.pop("name", None)
    copy.pop("description", None)
    extra = copy.pop("extra", {})

    model = acp_models.Metadata.model_validate(copy)
    if extra:
        for k, v in extra.items():
            setattr(model, k, v)
    return model


def _react_agent_factory(agent: ReActAgent, *, metadata: AcpAgentServerMetadata | None = None) -> AcpAgent:
    if metadata is None:
        metadata = {}

    async def run(
        input: list[acp_models.Message], context: acp_context.Context
    ) -> AsyncGenerator[acp_types.RunYield, acp_types.RunYieldResume]:
        framework_messages = [
            acp_msg_to_framework_msg(Role(message.parts[0].role), str(message))  # type: ignore[attr-defined]
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

    return AcpAgent(
        fn=run,
        name=metadata.get("name", agent.meta.name),
        description=metadata.get("description", agent.meta.description),
        metadata=to_acp_agent_metadata(metadata),
    )


AcpAgentServer.register_factory(ReActAgent, _react_agent_factory)


def _tool_calling_agent_factory(agent: ToolCallingAgent, *, metadata: AcpAgentServerMetadata | None = None) -> AcpAgent:
    if metadata is None:
        metadata = {}

    async def run(
        input: list[acp_models.Message], context: acp_context.Context
    ) -> AsyncGenerator[acp_types.RunYield, acp_types.RunYieldResume]:
        framework_messages = [
            acp_msg_to_framework_msg(Role(message.parts[0].role), str(message))  # type: ignore[attr-defined]
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

    return AcpAgent(
        fn=run,
        name=metadata.get("name", agent.meta.name),
        description=metadata.get("description", agent.meta.description),
        metadata=to_acp_agent_metadata(metadata),
    )


AcpAgentServer.register_factory(ToolCallingAgent, _tool_calling_agent_factory)
