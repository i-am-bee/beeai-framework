# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0
import contextlib
import uuid
from collections.abc import AsyncGenerator, Generator
from typing import Annotated, Self

from pydantic import BaseModel
from typing_extensions import TypedDict, TypeVar, Unpack, override

from beeai_framework.adapters.beeai_platform.serve.io import BeeAIPlatformIOContext
from beeai_framework.agents.experimental import RequirementAgent
from beeai_framework.agents.experimental.events import RequirementAgentSuccessEvent
from beeai_framework.agents.react import ReActAgent, ReActAgentUpdateEvent
from beeai_framework.agents.tool_calling import ToolCallingAgent, ToolCallingAgentSuccessEvent
from beeai_framework.backend import AssistantMessage, MessageTextContent, MessageToolCallContent, ToolMessage
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
        if not self._members:
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


def _react_agent_factory(
    agent: ReActAgent, *, metadata: BeeAIPlatformServerMetadata | None = None, memory_manager: MemoryManager
) -> beeai_agent.Agent:
    async def run(
        message: a2a_types.Message,
        context: beeai_context.RunContext,
        trajectory: Annotated[beeai_extensions.TrajectoryExtensionServer, beeai_extensions.TrajectoryExtensionSpec()],
        citation: Annotated[beeai_extensions.CitationExtensionServer, beeai_extensions.CitationExtensionSpec()],
    ) -> AsyncGenerator[beeai_types.RunYield, beeai_types.RunYieldResume]:
        cloned_agent = await agent.clone() if isinstance(agent, Cloneable) else agent
        await init_agent_memory(cloned_agent, memory_manager, context.context_id)

        with BeeAIPlatformIOContext(context):
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
) -> beeai_agent.Agent:
    async def run(
        message: a2a_types.Message,
        context: beeai_context.RunContext,
        trajectory: Annotated[beeai_extensions.TrajectoryExtensionServer, beeai_extensions.TrajectoryExtensionSpec()],
        citation: Annotated[beeai_extensions.CitationExtensionServer, beeai_extensions.CitationExtensionSpec()],
    ) -> AsyncGenerator[beeai_types.RunYield, beeai_types.RunYieldResume]:
        cloned_agent = await agent.clone() if isinstance(agent, Cloneable) else agent
        await init_agent_memory(cloned_agent, memory_manager, context.context_id)

        with BeeAIPlatformIOContext(context):
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
) -> beeai_agent.Agent:
    async def run(
        message: a2a_types.Message,
        context: beeai_context.RunContext,
        trajectory: Annotated[beeai_extensions.TrajectoryExtensionServer, beeai_extensions.TrajectoryExtensionSpec()],
        citation: Annotated[beeai_extensions.CitationExtensionServer, beeai_extensions.CitationExtensionSpec()],
    ) -> AsyncGenerator[beeai_types.RunYield, beeai_types.RunYieldResume]:
        cloned_agent = await agent.clone() if isinstance(agent, Cloneable) else agent
        await init_agent_memory(cloned_agent, memory_manager, context.context_id)

        with BeeAIPlatformIOContext(context):
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
