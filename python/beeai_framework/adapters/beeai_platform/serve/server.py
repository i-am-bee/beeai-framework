# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0
import contextlib
from collections.abc import AsyncGenerator, Sequence
from typing import Annotated, Self

from pydantic import BaseModel
from typing_extensions import TypedDict, TypeVar, Unpack, override

from beeai_framework.agents.experimental import RequirementAgent
from beeai_framework.agents.experimental.events import RequirementAgentSuccessEvent
from beeai_framework.agents.react import ReActAgent, ReActAgentUpdateEvent
from beeai_framework.agents.tool_calling import ToolCallingAgent, ToolCallingAgentSuccessEvent
from beeai_framework.serve.errors import FactoryAlreadyRegisteredError
from beeai_framework.utils.lists import find_index

try:
    import a2a.types as a2a_types
    import beeai_sdk.a2a.extensions as beeai_extensions
    import beeai_sdk.a2a.types as beeai_types
    import beeai_sdk.server as beeai_server
    import beeai_sdk.server.agent as beeai_agent
    import beeai_sdk.server.context as beeai_context
    from beeai_sdk.a2a.extensions.ui.agent_detail import AgentDetail

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


class BeeAIPlatformServerConfig(BaseModel):
    """Configuration for the BeeAIServer."""

    host: str = "127.0.0.1"
    port: int = 9999
    configure_telemetry: bool = True


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
        beeai_agent.Agent,
        BeeAIPlatformServerConfig,
    ],
):
    def __init__(
        self, *, config: ModelLike[BeeAIPlatformServerConfig] | None = None, memory_manager: MemoryManager | None = None
    ) -> None:
        super().__init__(
            config=to_model(BeeAIPlatformServerConfig, config or BeeAIPlatformServerConfig()),
            memory_manager=memory_manager,
        )
        self._metadata_by_agent: dict[AnyAgentLike, BeeAIPlatformServerMetadata] = {}
        self._server = beeai_server.Server()

    def _setup_member(self) -> None:
        if len(self._members) == 0:
            raise ValueError("No agents registered to the server.")

        member = self._members[0]
        factory = type(self)._factories[type(member)]
        config = self._metadata_by_agent.get(member, BeeAIPlatformServerConfig())
        self._server._agent = factory(member, metadata=config, memory_manager=self._memory_manager)  # type: ignore[call-arg]

    def serve(self) -> None:
        self._setup_member()
        self._server.run(**self._config.model_dump(exclude_none=True))

    async def aserve(self) -> None:
        self._setup_member()
        await self._server.serve(**self._config.model_dump(exclude_none=True))

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

    @override
    def register_many(self, input: Sequence[AnyAgentLike]) -> Self:
        raise NotImplementedError("register_many is not implemented for BeeAIPlatformServer")


def _react_agent_factory(
    agent: ReActAgent, *, metadata: BeeAIPlatformServerMetadata | None = None, memory_manager: MemoryManager
) -> beeai_agent.Agent:
    async def run(
        message: a2a_types.Message,
        context: beeai_context.RunContext,
        trajectory: Annotated[beeai_extensions.TrajectoryExtensionServer, beeai_extensions.TrajectoryExtensionSpec()],
        citation: Annotated[beeai_extensions.CitationExtensionServer, beeai_extensions.CitationExtensionSpec()],
    ) -> AsyncGenerator[beeai_types.RunYield, beeai_types.RunYieldResume]:
        await init_agent_memory(agent, memory_manager, context.context_id)
        await agent.memory.add(convert_a2a_to_framework_message(message))

        async for data, event in agent.run():
            match (data, event.name):
                case (ReActAgentUpdateEvent(), "partial_update"):
                    update = data.update.value
                    update = update.get_text_content() if hasattr(update, "get_text_content") else str(update)
                    match data.update.key:
                        case "thought" | "tool_name" | "tool_input" | "tool_output":
                            yield trajectory.trajectory_metadata(title=data.update.key, content=update)
                        case "final_answer":
                            yield beeai_types.AgentMessage(text=update)

    metadata = metadata or {}
    return beeai_agent.agent(**metadata)(run)


with contextlib.suppress(FactoryAlreadyRegisteredError):
    BeeAIPlatformServer.register_factory(ReActAgent, _react_agent_factory)  # type: ignore[arg-type]


def _tool_calling_agent_factory(
    agent: ToolCallingAgent, *, metadata: BeeAIPlatformServerMetadata | None = None, memory_manager: MemoryManager
) -> beeai_agent.Agent:
    async def run(
        message: a2a_types.Message,
        context: beeai_context.RunContext,
        trajectory: Annotated[beeai_extensions.TrajectoryExtensionServer, beeai_extensions.TrajectoryExtensionSpec()],
        citation: Annotated[beeai_extensions.CitationExtensionServer, beeai_extensions.CitationExtensionSpec()],
    ) -> AsyncGenerator[beeai_types.RunYield, beeai_types.RunYieldResume]:
        await init_agent_memory(agent, memory_manager, context.context_id)
        await agent.memory.add(convert_a2a_to_framework_message(message))

        last_msg: AnyMessage | None = None
        async for data, _ in agent.run():
            messages = data.state.memory.messages
            if last_msg is None:
                last_msg = messages[-1]

            cur_index = find_index(messages, lambda msg: msg is last_msg, fallback=-1, reverse_traversal=True)  # noqa: B023
            for msg in messages[cur_index + 1 :]:
                yield trajectory.trajectory_metadata(title="message", content=msg.text)
                last_msg = msg

            if isinstance(data, ToolCallingAgentSuccessEvent) and data.state.result is not None:
                yield beeai_types.AgentMessage(text=data.state.result.text)

    metadata = metadata or {}
    return beeai_agent.agent(**metadata)(run)


with contextlib.suppress(FactoryAlreadyRegisteredError):
    BeeAIPlatformServer.register_factory(ToolCallingAgent, _tool_calling_agent_factory)  # type: ignore[arg-type]


def _requirement_agent_factory(
    agent: RequirementAgent, *, metadata: BeeAIPlatformServerMetadata | None = None, memory_manager: MemoryManager
) -> beeai_agent.Agent:
    async def run(
        message: a2a_types.Message,
        context: beeai_context.RunContext,
        trajectory: Annotated[beeai_extensions.TrajectoryExtensionServer, beeai_extensions.TrajectoryExtensionSpec()],
        citation: Annotated[beeai_extensions.CitationExtensionServer, beeai_extensions.CitationExtensionSpec()],
    ) -> AsyncGenerator[beeai_types.RunYield, beeai_types.RunYieldResume]:
        await init_agent_memory(agent, memory_manager, context.context_id)
        await agent.memory.add(convert_a2a_to_framework_message(message))

        last_msg: AnyMessage | None = None
        async for data, _ in agent.run():
            messages = data.state.memory.messages
            if last_msg is None:
                last_msg = messages[-1]

            cur_index = find_index(messages, lambda msg: msg is last_msg, fallback=-1, reverse_traversal=True)  # noqa: B023
            for msg in messages[cur_index + 1 :]:
                yield trajectory.trajectory_metadata(title="message", content=msg.text)
                last_msg = msg

            if isinstance(data, RequirementAgentSuccessEvent) and data.state.answer is not None:
                yield beeai_types.AgentMessage(text=data.state.answer.text)

    metadata = metadata or {}
    return beeai_agent.agent(**metadata)(run)


with contextlib.suppress(FactoryAlreadyRegisteredError):
    BeeAIPlatformServer.register_factory(RequirementAgent, _requirement_agent_factory)  # type: ignore[arg-type]
