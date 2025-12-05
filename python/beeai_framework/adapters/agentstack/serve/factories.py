# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0
import uuid
from collections.abc import AsyncGenerator
from typing import Annotated, Any, Unpack, cast

from beeai_framework.adapters.agentstack.backend.chat import AgentStackChatModel
from beeai_framework.adapters.agentstack.context import AgentStackContext
from beeai_framework.adapters.agentstack.serve.server import (
    AgentStackMemoryManager,
    AgentStackServerMetadata,
    AgentStackSettingsContent,
    BaseAgentStackServerMetadata,
)
from beeai_framework.adapters.agentstack.serve.types import BaseAgentStackExtensions
from beeai_framework.adapters.agentstack.serve.utils import init_agent_stack_memory
from beeai_framework.agents import BaseAgent
from beeai_framework.agents.react import ReActAgent, ReActAgentUpdateEvent
from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.agents.requirement.events import RequirementAgentFinalAnswerEvent
from beeai_framework.agents.requirement.utils._tool import FinalAnswerTool
from beeai_framework.agents.tool_calling import ToolCallingAgent
from beeai_framework.emitter import EventMeta
from beeai_framework.logger import Logger
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.middleware.trajectory import (
    GlobalTrajectoryMiddleware,
    GlobalTrajectoryMiddlewareErrorEvent,
    GlobalTrajectoryMiddlewareStartEvent,
    GlobalTrajectoryMiddlewareSuccessEvent,
)
from beeai_framework.runnable import Runnable
from beeai_framework.tools import AnyTool, Tool, ToolOutput
from beeai_framework.utils.cloneable import Cloneable, clone_class
from beeai_framework.utils.lists import remove_falsy
from beeai_framework.utils.strings import to_json

try:
    import a2a.types as a2a_types
    import agentstack_sdk.a2a.extensions as agentstack_extensions
    import agentstack_sdk.a2a.types as agentstack_types
    import agentstack_sdk.server.agent as agentstack_agent
    import agentstack_sdk.server.context as agentstack_context

    from beeai_framework.adapters.a2a.agents._utils import convert_a2a_to_framework_message, convert_to_a2a_message
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [agentstack] not found.\nRun 'pip install \"beeai-framework[agentstack]\"' to install."
    ) from e

from beeai_framework.serve import MemoryManager

logger = Logger(__name__)


def _react_agent_factory(
    agent: ReActAgent, *, metadata: AgentStackServerMetadata | None = None, memory_manager: MemoryManager
) -> agentstack_agent.AgentFactory:
    agent_metadata, extensions = _init_metadata(agent, metadata)

    llm = agent._input.llm
    if isinstance(llm, AgentStackChatModel):
        extensions.__annotations__["llm_ext"] = Annotated[
            agentstack_extensions.LLMServiceExtensionServer,
            agentstack_extensions.LLMServiceExtensionSpec.single_demand(suggested=tuple(llm.preferred_models)),
        ]

    async def run(
        message: a2a_types.Message,
        context: agentstack_context.RunContext,
        **extra_extensions: Unpack[extensions],  # type: ignore
    ) -> AsyncGenerator[agentstack_types.RunYield, agentstack_types.RunYieldResume]:
        cloned_agent = await agent.clone() if isinstance(agent, Cloneable) else agent
        await init_agent_stack_memory(cloned_agent, memory_manager, context)

        # pyrefly: ignore [bad-argument-type]
        has_tool_settings, allowed_tools = _get_tools_settings(extra_extensions.get("settings"))
        if has_tool_settings:
            cloned_agent._input.tools = [tool for tool in cloned_agent._input.tools if tool.name in allowed_tools]

        with AgentStackContext(
            context,
            metadata=message.metadata,
            # pyrefly: ignore [bad-argument-type]
            llm=extra_extensions.get("llm_ext"),
            extra_extensions=extra_extensions,  # type: ignore[arg-type]
        ) as stack_context:
            artifact_id = uuid.uuid4()
            append = False

            @cloned_agent.emitter.on("partial_update")
            async def on_partial_update(data: ReActAgentUpdateEvent, _: EventMeta) -> None:
                nonlocal append
                if data.update.key == "final_answer":
                    update = data.update.value
                    update = update.get_text_content() if hasattr(update, "get_text_content") else str(update)
                    await _send_final_answer_update(context, artifact_id, update, append=append)
                    append = True

            @cloned_agent.emitter.on("update")
            async def on_update(data: ReActAgentUpdateEvent, _: EventMeta) -> None:
                if data.update.key == "thought":
                    update = data.update.parsed_value
                    update = update.get_text_content() if hasattr(update, "get_text_content") else str(update)
                    await context.yield_async(
                        extra_extensions["trajectory"].trajectory_metadata(title=data.update.key, content=update)
                    )

            result = await cloned_agent.run([convert_a2a_to_framework_message(message)]).middleware(
                create_tool_trajectory_middleware(stack_context)
            )

            agent_response = convert_to_a2a_message(result.last_message)
            if isinstance(memory_manager, AgentStackMemoryManager):
                await context.store(message)
                await context.store(agent_response)

            if append:
                await _send_final_answer_update(context, artifact_id, last_chunk=True)
            else:
                yield agent_response

    return agentstack_agent.agent(**agent_metadata)(run)


def _tool_calling_agent_factory(
    agent: ToolCallingAgent, *, metadata: AgentStackServerMetadata | None = None, memory_manager: MemoryManager
) -> agentstack_agent.AgentFactory:
    agent_metadata, extensions = _init_metadata(agent, metadata)

    llm = agent._llm
    if isinstance(llm, AgentStackChatModel):
        extensions.__annotations__["llm_ext"] = Annotated[
            agentstack_extensions.LLMServiceExtensionServer,
            agentstack_extensions.LLMServiceExtensionSpec.single_demand(suggested=tuple(llm.preferred_models)),
        ]

    async def run(
        message: a2a_types.Message,
        context: agentstack_context.RunContext,
        **extra_extensions: Unpack[extensions],  # type: ignore
    ) -> AsyncGenerator[agentstack_types.RunYield, agentstack_types.RunYieldResume]:
        cloned_agent = await agent.clone() if isinstance(agent, Cloneable) else agent
        await init_agent_stack_memory(cloned_agent, memory_manager, context)

        # pyrefly: ignore [bad-argument-type]
        has_tool_settings, allowed_tools = _get_tools_settings(extra_extensions.get("settings"))
        if has_tool_settings:
            cloned_agent._tools = [tool for tool in cloned_agent._tools if tool.name in allowed_tools]

        with AgentStackContext(
            context,
            metadata=message.metadata,
            # pyrefly: ignore [bad-argument-type]
            llm=extra_extensions.get("llm_ext"),
            extra_extensions=extra_extensions,  # type: ignore[arg-type]
        ) as stack_context:
            result = await cloned_agent.run([convert_a2a_to_framework_message(message)]).middleware(
                create_tool_trajectory_middleware(stack_context)
            )

            agent_response = convert_to_a2a_message(result.last_message)
            if isinstance(memory_manager, AgentStackMemoryManager):
                await context.store(message)
                await context.store(agent_response)

            yield agent_response

    return agentstack_agent.agent(**agent_metadata)(run)


def _requirement_agent_factory(
    agent: RequirementAgent, *, metadata: AgentStackServerMetadata | None = None, memory_manager: MemoryManager
) -> agentstack_agent.AgentFactory:
    agent_metadata, extensions = _init_metadata(agent, metadata)

    llm = agent._llm
    if isinstance(llm, AgentStackChatModel):
        extensions.__annotations__["llm_ext"] = Annotated[
            agentstack_extensions.LLMServiceExtensionServer,
            agentstack_extensions.LLMServiceExtensionSpec.single_demand(suggested=tuple(llm.preferred_models)),
        ]

    async def run(
        message: a2a_types.Message,
        context: agentstack_context.RunContext,
        **extra_extensions: Unpack[extensions],  # type: ignore
    ) -> AsyncGenerator[agentstack_types.RunYield, agentstack_types.RunYieldResume]:
        cloned_agent = await agent.clone() if isinstance(agent, Cloneable) else agent
        await init_agent_stack_memory(cloned_agent, memory_manager, context)

        # pyrefly: ignore [bad-argument-type]
        has_tool_settings, allowed_tools = _get_tools_settings(extra_extensions.get("settings"))
        if has_tool_settings:
            logger.warning("Tools settings is ignored for the RequirementAgent")

        with AgentStackContext(
            context,
            metadata=message.metadata,
            # pyrefly: ignore [bad-argument-type]
            llm=extra_extensions.get("llm_ext"),
            extra_extensions=extra_extensions,  # type: ignore[arg-type]
        ) as stack_context:
            artifact_id = uuid.uuid4()
            append = False

            @cloned_agent.emitter.on("final_answer")
            async def on_final_answer(data: RequirementAgentFinalAnswerEvent, _: EventMeta) -> None:
                nonlocal append
                await _send_final_answer_update(context, artifact_id, data.delta, append=append)
                append = True

            result = await cloned_agent.run([convert_a2a_to_framework_message(message)]).middleware(
                create_tool_trajectory_middleware(stack_context)
            )

            agent_response = convert_to_a2a_message(result.last_message)
            if isinstance(memory_manager, AgentStackMemoryManager):
                await context.store(message)
                await context.store(agent_response)

            if append:
                await _send_final_answer_update(context, artifact_id, last_chunk=True)
            else:
                yield agent_response

    return agentstack_agent.agent(**agent_metadata)(run)


async def _send_final_answer_update(
    context: agentstack_context.RunContext,
    artifact_id: uuid.UUID,
    update: str = "",
    *,
    append: bool = True,
    last_chunk: bool = False,
) -> None:
    await context.yield_async(
        a2a_types.TaskArtifactUpdateEvent(
            append=append,
            context_id=context.context_id,
            task_id=context.task_id,
            last_chunk=last_chunk,
            artifact=a2a_types.Artifact(
                name="final_answer",
                artifact_id=str(artifact_id),
                parts=[a2a_types.Part(root=a2a_types.TextPart(text=update))],
            ),
        )
    )


def _runnable_factory(
    runnable: Runnable[Any], *, metadata: AgentStackServerMetadata | None = None, memory_manager: MemoryManager
) -> agentstack_agent.AgentFactory:
    runnable_metadata, extensions = _init_metadata(runnable, metadata)

    async def run(
        message: a2a_types.Message,
        context: agentstack_context.RunContext,
        **extra_extensions: Unpack[extensions],  # type: ignore
    ) -> AsyncGenerator[agentstack_types.RunYield, agentstack_types.RunYieldResume]:
        cloned_runnable = await runnable.clone() if isinstance(runnable, Cloneable) else runnable
        memory = None
        if isinstance(memory_manager, AgentStackMemoryManager):
            history = [msg async for msg in context.load_history() if msg.parts]
            messages = [convert_a2a_to_framework_message(msg) for msg in history]
        else:
            try:
                memory = await memory_manager.get(context.context_id)
            except KeyError:
                memory = UnconstrainedMemory()
                await memory_manager.set(context.context_id, memory)

            await memory.add(convert_a2a_to_framework_message(message))
            messages = memory.messages

        with AgentStackContext(
            context,
            metadata=message.metadata,
            # pyrefly: ignore [bad-argument-type]
            llm=extra_extensions.get("llm_ext"),
            extra_extensions=extra_extensions,  # type: ignore[arg-type]
        ):
            # pyrefly: ignore [missing-attribute]
            data = await cloned_runnable.run(messages)
            if memory is not None:
                await memory.add(data.last_message)

            agent_response = agentstack_types.AgentMessage(
                text=data.last_message.text,
                context_id=context.context_id,
                task_id=context.task_id,
                reference_task_ids=[task.id for task in (context.related_tasks or [])],
            )
            if isinstance(memory_manager, AgentStackMemoryManager):
                await context.store(message)
                await context.store(agent_response)

            yield agent_response

    return agentstack_agent.agent(**runnable_metadata)(run)


def _get_tools_settings(
    settings: Annotated[agentstack_extensions.SettingsExtensionServer, Any] | None,
) -> tuple[bool, list[str]]:
    if settings:
        try:
            parsed_settings = settings.parse_settings_response()

            tools_settings = parsed_settings.values.get("tools")
            if tools_settings and tools_settings.type == "checkbox_group":
                return True, [key for key, value in tools_settings.values.items() if value.value]

        except Exception:
            logger.exception("Failed to parse settings response")
    return False, []


def _init_metadata(
    runnable: Runnable[Any],
    base: AgentStackServerMetadata | None = None,
) -> tuple[BaseAgentStackServerMetadata, type[BaseAgentStackExtensions]]:
    base_copy: AgentStackServerMetadata = base.copy() if base else AgentStackServerMetadata()
    base_extension: type[BaseAgentStackExtensions] = base_copy.pop("extensions", BaseAgentStackExtensions)
    extensions = clone_class(base_extension)
    settings: set[AgentStackSettingsContent] = base_copy.pop("settings", set())

    if settings and isinstance(runnable, ToolCallingAgent | ReActAgent):
        tools = runnable.meta.tools if isinstance(runnable, ReActAgent) else runnable._tools
        extensions.__annotations__["settings"] = Annotated[
            agentstack_extensions.SettingsExtensionServer,
            agentstack_extensions.SettingsExtensionSpec(
                params=agentstack_extensions.SettingsRender(
                    fields=remove_falsy(
                        [
                            agentstack_extensions.CheckboxGroupField(
                                id=AgentStackSettingsContent.TOOLS,
                                fields=[
                                    agentstack_extensions.ui.settings.CheckboxField(
                                        id=tool.name,
                                        label=tool.name,
                                        default_value=True,
                                    )
                                    for tool in tools
                                ],
                            )
                            if AgentStackSettingsContent.TOOLS in settings and tools
                            else None
                        ]
                    ),
                ),
            ),
        ]

    metadata = BaseAgentStackServerMetadata(**base_copy)  # type: ignore
    if isinstance(runnable, BaseAgent):
        if not metadata.get("name"):
            metadata["name"] = runnable.meta.name
        if not metadata.get("description"):
            metadata["description"] = runnable.meta.description

    return metadata, extensions


def create_tool_trajectory_middleware(
    agent_stack_context: AgentStackContext,
) -> GlobalTrajectoryMiddleware:
    context = agent_stack_context.context
    trajectory = agent_stack_context.extensions["trajectory"]
    tool_calls_trajectory_middleware = GlobalTrajectoryMiddleware(
        included=[Tool], excluded=[FinalAnswerTool], target=False, match_nested=True
    )

    @tool_calls_trajectory_middleware.emitter.on("start")
    async def send_tool_call_start(data: GlobalTrajectoryMiddlewareStartEvent, _: EventMeta) -> None:
        tool_start_event, tool_start_meta = data.origin
        tool = cast(AnyTool, tool_start_meta.creator.instance)  # type: ignore[attr-defined]
        if _check_is_final_answer(tool):
            return
        await context.yield_async(
            trajectory.trajectory_metadata(
                title=f"{'--> ' * (data.level.relative - 1)}{tool.name} (request)",
                content=to_json(tool_start_event.input, sort_keys=False, indent=4),
            )
        )

    @tool_calls_trajectory_middleware.emitter.on("success")
    async def send_tool_call_success(data: GlobalTrajectoryMiddlewareSuccessEvent, _: EventMeta) -> None:
        tool_success_event, tool_success_meta = data.origin
        tool_output = cast(ToolOutput, tool_success_event.output)
        tool = cast(AnyTool, tool_success_meta.creator.instance)  # type: ignore[attr-defined]
        if _check_is_final_answer(tool):
            return
        await context.yield_async(
            trajectory.trajectory_metadata(
                title=f"{'<-- ' * (data.level.relative - 1)}{tool.name} (response)",
                content=tool_output.get_text_content(),
            )
        )

    @tool_calls_trajectory_middleware.emitter.on("error")
    async def send_tool_call_error(data: GlobalTrajectoryMiddlewareErrorEvent, _: EventMeta) -> None:
        tool_error_event, tool_error_meta = data.origin
        tool = cast(AnyTool, tool_error_meta.creator.instance)  # type: ignore[attr-defined]
        await context.yield_async(
            trajectory.trajectory_metadata(
                title=f"{'<-- ' * (data.level.relative - 1)}{tool.name} (error)", content=tool_error_event.explain()
            )
        )

    return tool_calls_trajectory_middleware


def _check_is_final_answer(tool: AnyTool) -> bool:
    return isinstance(tool, FinalAnswerTool) or tool.name == "final_answer"
