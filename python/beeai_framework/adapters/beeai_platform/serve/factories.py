# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import uuid
from collections.abc import AsyncGenerator
from typing import Annotated, Any

from beeai_framework.adapters.beeai_platform.backend.chat import BeeAIPlatformChatModel
from beeai_framework.adapters.beeai_platform.context import BeeAIPlatformContext
from beeai_framework.adapters.beeai_platform.serve.server import BeeAIPlatformMemoryManager, BeeAIPlatformServerMetadata
from beeai_framework.adapters.beeai_platform.serve.utils import init_beeai_platform_memory, send_message_trajectory
from beeai_framework.agents import AnyAgent
from beeai_framework.agents.react import ReActAgent, ReActAgentUpdateEvent
from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.agents.requirement.events import RequirementAgentSuccessEvent
from beeai_framework.agents.tool_calling import ToolCallingAgent, ToolCallingAgentSuccessEvent
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.runnable import Runnable
from beeai_framework.utils.cloneable import Cloneable
from beeai_framework.utils.lists import find_index

try:
    import a2a.types as a2a_types
    import beeai_sdk.a2a.extensions as beeai_extensions
    import beeai_sdk.a2a.types as beeai_types
    import beeai_sdk.server.agent as beeai_agent
    import beeai_sdk.server.context as beeai_context

    from beeai_framework.adapters.a2a.agents._utils import convert_a2a_to_framework_message
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [beeai-platform] not found.\nRun 'pip install \"beeai-framework[beeai-platform]\"' to install."
    ) from e

from beeai_framework.backend.message import AnyMessage
from beeai_framework.serve import MemoryManager


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


def _runnable_factory(
    runnable: Runnable[Any], *, metadata: BeeAIPlatformServerMetadata | None = None, memory_manager: MemoryManager
) -> beeai_agent.AgentFactory:
    async def run(
        message: a2a_types.Message,
        context: beeai_context.RunContext,
    ) -> AsyncGenerator[beeai_types.RunYield, beeai_types.RunYieldResume]:
        cloned_runnable = await runnable.clone() if isinstance(runnable, Cloneable) else runnable
        memory = None
        if isinstance(memory_manager, BeeAIPlatformMemoryManager):
            history = [msg async for msg in context.store.load_history() if msg.parts]
            messages = [convert_a2a_to_framework_message(msg) for msg in history]
        else:
            try:
                memory = await memory_manager.get(context.context_id)
            except KeyError:
                memory = UnconstrainedMemory()
                await memory_manager.set(context.context_id, memory)

            await memory.add(convert_a2a_to_framework_message(message))
            messages = memory.messages

        data = await cloned_runnable.run(messages)
        if memory is not None:
            await memory.add(data.last_message)
        yield beeai_types.AgentMessage(
            text=data.last_message.text,
            context_id=context.context_id,
            task_id=context.task_id,
            reference_task_ids=[task.id for task in (context.related_tasks or [])],
        )

    return beeai_agent.agent(**(metadata or {}))(run)


def _init_metadata(
    agent: AnyAgent,
    base: BeeAIPlatformServerMetadata | None = None,
) -> BeeAIPlatformServerMetadata:
    copy = (base or {}).copy()
    if not copy.get("name"):
        copy["name"] = agent.meta.name
    if not copy.get("description"):
        copy["description"] = agent.meta.description
    return copy
