# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
import os
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Generator
from datetime import timedelta
from typing import Annotated, Any, Self

import uvicorn
from pydantic import BaseModel
from typing_extensions import TypedDict, TypeVar, Unpack, override

from beeai_framework.adapters.beeai_platform.backend.chat import BeeAIPlatformChatModel
from beeai_framework.adapters.beeai_platform.context import BeeAIPlatformContext
from beeai_framework.agents.experimental import RequirementAgent
from beeai_framework.agents.experimental.events import RequirementAgentSuccessEvent
from beeai_framework.agents.react import ReActAgent, ReActAgentUpdateEvent
from beeai_framework.agents.tool_calling import ToolCallingAgent, ToolCallingAgentSuccessEvent
from beeai_framework.backend import AssistantMessage, MessageTextContent, MessageToolCallContent, ToolMessage
from beeai_framework.memory import BaseMemory
from beeai_framework.serve.errors import FactoryAlreadyRegisteredError
from beeai_framework.utils.cloneable import Cloneable
from beeai_framework.utils.lists import find_index

try:
    import a2a.types as a2a_types
    import beeai_sdk.a2a.extensions as beeai_extensions
    import beeai_sdk.a2a.types as beeai_types
    import beeai_sdk.server as beeai_server
    import beeai_sdk.server.agent as beeai_agent
    import beeai_sdk.server.context as beeai_context
    import beeai_sdk.server.store.context_store as beeai_context_store
    import beeai_sdk.server.store.platform_context_store as beeai_platform_context_store
    from beeai_sdk.a2a.extensions.ui.agent_detail import AgentDetail
    from beeai_sdk.server.dependencies import Dependency

    from beeai_framework.adapters.a2a.agents._utils import convert_a2a_to_framework_message
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [beeai-platform] not found.\nRun 'pip install \"beeai-framework[beeai-platform]\"' to install."
    ) from e

from beeai_framework.agents import AnyAgent
from beeai_framework.backend.message import AnyMessage
from beeai_framework.serve import MemoryManager, Server, init_agent_memory
from beeai_framework.utils.models import ModelLike, to_model

AnyAgentLike = TypeVar("AnyAgentLike", bound=AnyAgent, default=AnyAgent)


class BeeAIPlatformMemoryManager(MemoryManager):
    async def set(self, key: str, value: BaseMemory) -> None:
        pass

    async def get(self, key: str) -> BaseMemory:  # type: ignore[empty-body]
        pass

    async def contains(self, key: str) -> bool:  # type: ignore[empty-body]
        pass


class DummyContextStoreInstance(beeai_context_store.ContextStoreInstance):
    async def load_history(self) -> AsyncIterator[a2a_types.Message | a2a_types.Artifact]:  # type: ignore
        pass

    async def store(self, data: a2a_types.Message | a2a_types.Artifact) -> None:
        pass


class DummyContextStore(beeai_context_store.ContextStore):
    _cs = DummyContextStoreInstance()

    async def create(
        self,
        context_id: str,
        initialized_dependencies: list[Dependency],  # type: ignore[type-arg]
    ) -> beeai_context_store.ContextStoreInstance:
        return self._cs


class BeeAIPlatformServerConfig(BaseModel):
    """Configuration for the BeeAIServer."""

    host: str = "127.0.0.1"
    port: int = 9999
    configure_telemetry: bool = True

    configure_logger: bool | None = None
    self_registration: bool | None = True
    run_limit: int | None = None
    run_ttl: timedelta | None = None
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


class BeeAIPlatformServerMetadata(TypedDict, total=False):
    name: str
    description: str
    additional_interfaces: list[a2a_types.AgentInterface]
    capabilities: a2a_types.AgentCapabilities
    default_input_modes: list[str]
    default_output_modes: list[str]
    detail: AgentDetail
    documentation_url: str
    icon_url: str
    preferred_transport: str
    provider: a2a_types.AgentProvider
    security: list[dict[str, list[str]]]
    security_schemes: dict[str, a2a_types.SecurityScheme]
    skills: list[a2a_types.AgentSkill]
    supports_authenticated_extended_card: bool
    version: str


class BeeAIPlatformServer(
    Server[
        AnyAgentLike,
        beeai_agent.AgentFactory,
        BeeAIPlatformServerConfig,
    ],
):
    def __init__(
        self, *, config: ModelLike[BeeAIPlatformServerConfig] | None = None, memory_manager: MemoryManager | None = None
    ) -> None:
        super().__init__(
            config=to_model(BeeAIPlatformServerConfig, config or BeeAIPlatformServerConfig()),
            memory_manager=memory_manager or BeeAIPlatformMemoryManager(),
        )
        self._metadata_by_agent: dict[AnyAgentLike, BeeAIPlatformServerMetadata] = {}
        self._server = beeai_server.Server()

    def _setup_member(self) -> beeai_context_store.ContextStore:
        if not self._members:
            raise ValueError("No agents registered to the server.")

        member = self._members[0]
        factory = type(self)._factories[type(member)]
        config = self._metadata_by_agent.get(member, BeeAIPlatformServerMetadata())
        self._server._agent_factory = factory(member, metadata=config, memory_manager=self._memory_manager)  # type: ignore[call-arg]
        return (
            beeai_platform_context_store.PlatformContextStore()
            if isinstance(self._memory_manager, BeeAIPlatformMemoryManager)
            else DummyContextStore()
        )

    def serve(self) -> None:
        context_store = self._setup_member()
        with contextlib.suppress(KeyboardInterrupt):
            self._server.run(
                **self._config.model_dump(exclude_none=True, exclude={"context_store": True}),
                context_store=context_store,
            )

    async def aserve(self) -> None:
        context_store = self._setup_member()
        with contextlib.suppress(KeyboardInterrupt):
            await self._server.serve(
                **self._config.model_dump(exclude_none=True, exclude={"context_store": True}),
                context_store=context_store,
            )

    @override
    def register(self, input: AnyAgentLike, **metadata: Unpack[BeeAIPlatformServerMetadata]) -> Self:
        if len(self._members) != 0:
            raise ValueError("BeeAIPlatformServer only supports one agent.")
        else:
            super().register(input)
            metadata = metadata or BeeAIPlatformServerMetadata()
            detail = metadata.setdefault("detail", AgentDetail())
            detail.framework = detail.framework or "BeeAI"

            self._metadata_by_agent[input] = metadata
            return self


def _react_agent_factory(
    agent: ReActAgent, *, metadata: BeeAIPlatformServerMetadata | None = None, memory_manager: MemoryManager
) -> beeai_agent.AgentFactory:
    llm = agent._input.llm
    preferred_models = llm.preferred_models if isinstance(llm, BeeAIPlatformChatModel) else []

    async def run(
        message: a2a_types.Message,
        context: beeai_context.RunContext,
        trajectory: Annotated[beeai_extensions.TrajectoryExtensionServer, beeai_extensions.TrajectoryExtensionSpec()],
        form: Annotated[beeai_extensions.FormExtensionServer, beeai_extensions.FormExtensionSpec(params=None)],
        llm_ext: Annotated[
            beeai_extensions.LLMServiceExtensionServer,
            beeai_extensions.LLMServiceExtensionSpec.single_demand(suggested=tuple(preferred_models)),
        ],
    ) -> AsyncGenerator[beeai_types.RunYield, beeai_types.RunYieldResume]:
        cloned_agent = await agent.clone() if isinstance(agent, Cloneable) else agent
        await init_beeai_platform_memory(cloned_agent, memory_manager, context)

        with BeeAIPlatformContext(context, form=form, llm=llm_ext):
            artifact_id = uuid.uuid4()
            append = False
            last_key = None
            last_update = None
            async for data, event in cloned_agent.run([convert_a2a_to_framework_message(message)]):
                match (data, event.name):
                    case (ReActAgentUpdateEvent(), "partial_update"):
                        match data.update.key:
                            case "thought" | "tool_name" | "tool_input" | "tool_output":
                                update = data.update.parsed_value
                                update = (
                                    update.get_text_content() if hasattr(update, "get_text_content") else str(update)
                                )
                                if last_key and last_key != data.update.key:
                                    yield trajectory.trajectory_metadata(title=last_key, content=last_update)
                                last_key = data.update.key
                                last_update = update
                            case "final_answer":
                                update = data.update.value
                                update = (
                                    update.get_text_content() if hasattr(update, "get_text_content") else str(update)
                                )
                                yield a2a_types.TaskArtifactUpdateEvent(
                                    append=append,
                                    context_id=context.context_id,
                                    task_id=context.task_id,
                                    last_chunk=False,
                                    artifact=a2a_types.Artifact(
                                        name="final_answer",
                                        artifact_id=str(artifact_id),
                                        parts=[a2a_types.Part(root=a2a_types.TextPart(text=update))],
                                    ),
                                )
                                append = True

            yield a2a_types.TaskArtifactUpdateEvent(
                append=True,
                context_id=context.context_id,
                task_id=context.task_id,
                last_chunk=True,
                artifact=a2a_types.Artifact(
                    name="final_answer",
                    artifact_id=str(artifact_id),
                    parts=[a2a_types.Part(root=a2a_types.TextPart(text=""))],
                ),
            )

    metadata = _init_metadata(agent, metadata)
    return beeai_agent.agent(**metadata)(run)


with contextlib.suppress(FactoryAlreadyRegisteredError):
    BeeAIPlatformServer.register_factory(ReActAgent, _react_agent_factory)  # type: ignore[arg-type]


def _tool_calling_agent_factory(
    agent: ToolCallingAgent, *, metadata: BeeAIPlatformServerMetadata | None = None, memory_manager: MemoryManager
) -> beeai_agent.AgentFactory:
    llm = agent._llm
    preferred_models = llm.preferred_models if isinstance(llm, BeeAIPlatformChatModel) else []

    async def run(
        message: a2a_types.Message,
        context: beeai_context.RunContext,
        trajectory: Annotated[beeai_extensions.TrajectoryExtensionServer, beeai_extensions.TrajectoryExtensionSpec()],
        form: Annotated[beeai_extensions.FormExtensionServer, beeai_extensions.FormExtensionSpec(params=None)],
        llm_ext: Annotated[
            beeai_extensions.LLMServiceExtensionServer,
            beeai_extensions.LLMServiceExtensionSpec.single_demand(suggested=tuple(preferred_models)),
        ],
    ) -> AsyncGenerator[beeai_types.RunYield, beeai_types.RunYieldResume]:
        cloned_agent = await agent.clone() if isinstance(agent, Cloneable) else agent
        await init_beeai_platform_memory(cloned_agent, memory_manager, context)

        with BeeAIPlatformContext(context, form=form, llm=llm_ext):
            last_msg: AnyMessage | None = None
            async for data, _ in cloned_agent.run([convert_a2a_to_framework_message(message)]):
                messages = data.state.memory.messages
                if last_msg is None:
                    last_msg = messages[-1]

                cur_index = find_index(messages, lambda msg: msg is last_msg, fallback=-1, reverse_traversal=True)  # noqa: B023
                for msg in messages[cur_index + 1 :]:
                    for value in send_message_trajectory(msg, trajectory):
                        yield value
                    last_msg = msg

                if isinstance(data, ToolCallingAgentSuccessEvent) and data.state.result is not None:
                    yield beeai_types.AgentMessage(text=data.state.result.text)

    metadata = _init_metadata(agent, metadata)
    return beeai_agent.agent(**metadata)(run)


with contextlib.suppress(FactoryAlreadyRegisteredError):
    BeeAIPlatformServer.register_factory(ToolCallingAgent, _tool_calling_agent_factory)  # type: ignore[arg-type]


def _requirement_agent_factory(
    agent: RequirementAgent, *, metadata: BeeAIPlatformServerMetadata | None = None, memory_manager: MemoryManager
) -> beeai_agent.AgentFactory:
    llm = agent._llm
    preferred_models = llm.preferred_models if isinstance(llm, BeeAIPlatformChatModel) else []

    async def run(
        message: a2a_types.Message,
        context: beeai_context.RunContext,
        trajectory: Annotated[beeai_extensions.TrajectoryExtensionServer, beeai_extensions.TrajectoryExtensionSpec()],
        form: Annotated[
            beeai_extensions.FormExtensionServer,
            beeai_extensions.FormExtensionSpec(params=None),
        ],
        llm_ext: Annotated[
            beeai_extensions.LLMServiceExtensionServer,
            beeai_extensions.LLMServiceExtensionSpec.single_demand(suggested=tuple(preferred_models)),
        ],
    ) -> AsyncGenerator[beeai_types.RunYield, beeai_types.RunYieldResume]:
        cloned_agent = await agent.clone() if isinstance(agent, Cloneable) else agent
        await init_beeai_platform_memory(cloned_agent, memory_manager, context)
        with BeeAIPlatformContext(context, form=form, llm=llm_ext):
            last_msg: AnyMessage | None = None
            async for data, _ in cloned_agent.run([convert_a2a_to_framework_message(message)]):
                messages = data.state.memory.messages
                if last_msg is None:
                    last_msg = messages[-1]

                cur_index = find_index(messages, lambda msg: msg is last_msg, fallback=-1, reverse_traversal=True)  # noqa: B023
                for msg in messages[cur_index + 1 :]:
                    for value in send_message_trajectory(msg, trajectory):
                        yield value
                    last_msg = msg

                if isinstance(data, RequirementAgentSuccessEvent) and data.state.answer is not None:
                    yield beeai_types.AgentMessage(text=data.state.answer.text)

    metadata = _init_metadata(agent, metadata)
    return beeai_agent.agent(**metadata)(run)


with contextlib.suppress(FactoryAlreadyRegisteredError):
    BeeAIPlatformServer.register_factory(RequirementAgent, _requirement_agent_factory)  # type: ignore[arg-type]


def send_message_trajectory(
    msg: AnyMessage,
    trajectory: Annotated[beeai_extensions.TrajectoryExtensionServer, beeai_extensions.TrajectoryExtensionSpec()],
) -> Generator[beeai_types.Metadata[str, beeai_extensions.Trajectory]]:
    if isinstance(msg, AssistantMessage):
        for content in msg.content:
            if isinstance(content, MessageTextContent):
                yield trajectory.trajectory_metadata(title="assistant", content=content.text)
            elif isinstance(content, MessageToolCallContent):
                if content.tool_name == "final_answer":
                    continue
                yield trajectory.trajectory_metadata(title=f"{content.tool_name} (request)", content=content.args)
    elif isinstance(msg, ToolMessage):
        for tool_call in msg.get_tool_results():
            if tool_call.tool_name == "final_answer":
                continue

            yield trajectory.trajectory_metadata(
                title=f"{tool_call.tool_name} (response)", content=str(tool_call.result)
            )


def _init_metadata(
    agent: AnyAgentLike,
    base: BeeAIPlatformServerMetadata | None = None,
) -> BeeAIPlatformServerMetadata:
    copy = (base or {}).copy()
    if not copy.get("name"):
        copy["name"] = agent.meta.name
    if not copy.get("description"):
        copy["description"] = agent.meta.description
    return copy


async def init_beeai_platform_memory(
    agent: AnyAgent, memory_manager: MemoryManager, context: beeai_context.RunContext
) -> None:
    if isinstance(memory_manager, BeeAIPlatformMemoryManager):
        history = [message async for message in context.store.load_history() if message.parts]
        agent.memory.reset()
        # last message is provided directly to the run method
        await agent.memory.add_many([convert_a2a_to_framework_message(message) for message in history[:-1]])
    else:
        await init_agent_memory(agent, memory_manager, context.context_id)
