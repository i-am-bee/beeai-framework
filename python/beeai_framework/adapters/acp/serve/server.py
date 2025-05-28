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

import asyncio
import os
from collections.abc import AsyncGenerator, Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any, Generic, Self

try:
    import acp_sdk.models as acp_models
    import acp_sdk.server.context as acp_context
    import acp_sdk.server.server as acp_server
    import acp_sdk.server.types as acp_types
    from acp_sdk import AnyModel, Author, Capability, Contributor, Dependency, Link
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [acp] not found.\nRun 'pip install \"beeai-framework[acp]\"' to install."
    ) from e

import uvicorn
from pydantic import BaseModel
from typing_extensions import TypedDict, TypeVar, Unpack, override

from beeai_framework.adapters.acp.serve._utils import acp_msg_to_framework_msg
from beeai_framework.adapters.acp.serve.agent import ACPServerAgent
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


class ACPServerMetadata(TypedDict, total=False):
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


class ACPServer(Generic[AnyAgentLike], Server[AnyAgentLike, ACPServerAgent, "ACPServerConfig"]):
    def __init__(self, *, config: ModelLike["ACPServerConfig"] | None = None) -> None:
        super().__init__(config=to_model(ACPServerConfig, config or {"self_registration": False}))
        self._metadata_by_agent: dict[AnyAgentLike, ACPServerMetadata] = {}
        self._server = acp_server.Server()

    def serve(self) -> None:
        for member in self.members:
            factory = type(self)._factories[type(member)]
            config = self._metadata_by_agent.get(member, None)
            self._server.register(factory(member, metadata=config))  # type: ignore[call-arg]

        self._server.run(**self._config.model_dump(exclude_none=True))

    @override
    def register(self, input: AnyAgentLike, **metadata: Unpack[ACPServerMetadata]) -> Self:
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


def to_acp_agent_metadata(metadata: ACPServerMetadata) -> acp_models.Metadata:
    copy = metadata.copy()
    copy.pop("name", None)
    copy.pop("description", None)
    extra = copy.pop("extra", {})

    model = acp_models.Metadata.model_validate(copy)
    if extra:
        for k, v in extra.items():
            setattr(model, k, v)
    return model


def _react_agent_factory(agent: ReActAgent, *, metadata: ACPServerMetadata | None = None) -> ACPServerAgent:
    if metadata is None:
        metadata = {}

    async def run(
        input: list[acp_models.Message], context: acp_context.Context
    ) -> AsyncGenerator[acp_types.RunYield, acp_types.RunYieldResume]:
        framework_messages = [
            acp_msg_to_framework_msg(Role(message.parts[0].role), str(message))  # type: ignore[attr-defined]
            for message in input
        ]

        agent.memory.reset()
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

    return ACPServerAgent(
        fn=run,
        name=metadata.get("name", agent.meta.name),
        description=metadata.get("description", agent.meta.description),
        metadata=to_acp_agent_metadata(metadata),
    )


ACPServer.register_factory(ReActAgent, _react_agent_factory)


def _tool_calling_agent_factory(
    agent: ToolCallingAgent, *, metadata: ACPServerMetadata | None = None
) -> ACPServerAgent:
    if metadata is None:
        metadata = {}

    async def run(
        input: list[acp_models.Message], context: acp_context.Context
    ) -> AsyncGenerator[acp_types.RunYield, acp_types.RunYieldResume]:
        framework_messages = [
            acp_msg_to_framework_msg(Role(message.parts[0].role), str(message))  # type: ignore[attr-defined]
            for message in input
        ]

        agent.memory.reset()
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

    return ACPServerAgent(
        fn=run,
        name=metadata.get("name", agent.meta.name),
        description=metadata.get("description", agent.meta.description),
        metadata=to_acp_agent_metadata(metadata),
    )


ACPServer.register_factory(ToolCallingAgent, _tool_calling_agent_factory)


class ACPServerConfig(BaseModel):
    """Configuration for the ACPServer."""

    configure_logger: bool | None = None
    configure_telemetry: bool | None = None
    self_registration: bool | None = False
    run_limit: int | None = None
    run_ttl: timedelta | None = None
    host: str | None = None
    port: int | None = None
    uds: str | None = None
    fd: int | None = None
    loop: uvicorn.config.LoopSetupType | None = None
    http: type[asyncio.Protocol] | uvicorn.config.HTTPProtocolType | None = None
    ws: type[asyncio.Protocol] | uvicorn.config.WSProtocolType | None = None
    ws_max_size: int | None = None
    ws_max_queue: int | None = None
    ws_ping_interval: float | None = None
    ws_ping_timeout: float | None = None
    ws_per_message_deflate: bool | None = None
    lifespan: uvicorn.config.LifespanType | None = None
    env_file: str | os.PathLike[str] | None = None
    log_config: dict[str, Any] | str | None = None
    log_level: str | int | None = None
    access_log: bool | None = None
    use_colors: bool | None = None
    interface: uvicorn.config.InterfaceType | None = None
    reload: bool | None = None
    reload_dirs: list[str] | str | None = None
    reload_delay: float | None = None
    reload_includes: list[str] | str | None = None
    reload_excludes: list[str] | str | None = None
    workers: int | None = None
    proxy_headers: bool | None = None
    server_header: bool | None = None
    date_header: bool | None = None
    forwarded_allow_ips: list[str] | str | None = None
    root_path: str | None = None
    limit_concurrency: int | None = None
    limit_max_requests: int | None = None
    backlog: int | None = None
    timeout_keep_alive: int | None = None
    timeout_notify: int | None = None
    timeout_graceful_shutdown: int | None = None
    callback_notify: Callable[..., Awaitable[None]] | None = None
    ssl_keyfile: str | os.PathLike[str] | None = None
    ssl_certfile: str | os.PathLike[str] | None = None
    ssl_keyfile_password: str | None = None
    ssl_version: int | None = None
    ssl_cert_reqs: int | None = None
    ssl_ca_certs: str | None = None
    ssl_ciphers: str | None = None
    headers: list[tuple[str, str]] | None = None
    factory: bool | None = None
    h11_max_incomplete_event_size: int | None = None
