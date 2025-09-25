# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing_extensions import TypeVar, override

from beeai_framework.adapters.a2a.agents._utils import convert_a2a_to_framework_message
from beeai_framework.agents.experimental.events import RequirementAgentStartEvent, RequirementAgentSuccessEvent
from beeai_framework.agents.react import ReActAgentUpdateEvent
from beeai_framework.agents.react.types import ReActAgentIterationResult
from beeai_framework.agents.tool_calling import ToolCallingAgentStartEvent, ToolCallingAgentSuccessEvent
from beeai_framework.backend import AssistantMessage, MessageToolCallContent, ToolMessage
from beeai_framework.emitter import Emitter, EventMeta
from beeai_framework.serve import MemoryManager, init_agent_memory
from beeai_framework.utils.cancellation import AbortController
from beeai_framework.utils.cloneable import Cloneable
from beeai_framework.utils.lists import find_index
from beeai_framework.utils.strings import to_json

try:
    import a2a.server as a2a_server
    import a2a.server.agent_execution as a2a_agent_execution
    import a2a.server.tasks as a2a_server_tasks
    import a2a.types as a2a_types
    import a2a.utils as a2a_utils
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [a2a] not found.\nRun 'pip install \"beeai-framework[a2a]\"' to install."
    ) from e

from beeai_framework.agents import AnyAgent
from beeai_framework.backend.message import (
    AnyMessage,
)
from beeai_framework.logger import Logger

AnyAgentLike = TypeVar("AnyAgentLike", bound=AnyAgent, default=AnyAgent)

logger = Logger(__name__)


class BaseA2AAgentExecutor(a2a_agent_execution.AgentExecutor):
    def __init__(
        self,
        agent: AnyAgentLike,
        agent_card: a2a_types.AgentCard,
        *,
        memory_manager: MemoryManager,
        send_trajectory: bool | None = True,
    ) -> None:
        super().__init__()
        self._agent = agent
        self.agent_card = agent_card
        self._abort_controller = AbortController()
        self._memory_manager = memory_manager
        self._send_trajectory = send_trajectory

    @override
    async def execute(
        self,
        context: a2a_agent_execution.RequestContext,
        event_queue: a2a_server.events.EventQueue,
    ) -> None:
        updater = a2a_server_tasks.TaskUpdater(event_queue, context.task_id, context.context_id)  # type: ignore[arg-type]
        if not context.current_task:
            if not context.message:
                raise ValueError("No message found in the request context.")
            context.current_task = a2a_utils.new_task(context.message)
            await updater.submit()

        cloned_agent = await self._agent.clone() if isinstance(self._agent, Cloneable) else self._agent
        await init_agent_memory(cloned_agent, self._memory_manager, context.context_id)
        new_messages = _extract_request_messages(context)

        await updater.start_work()
        try:
            response = await cloned_agent.run(new_messages, signal=self._abort_controller.signal).observe(
                lambda emitter: self._process_events(emitter, context, updater) if self._send_trajectory else ...
            )

            await updater.complete(
                a2a_utils.new_agent_text_message(
                    response.last_message.text,
                    context.context_id,
                    context.task_id,
                )
            )

        except Exception as e:
            logger.exception("Exception during execution")
            await updater.failed(
                message=a2a_utils.new_agent_text_message(str(e)),
            )

    @override
    async def cancel(
        self,
        context: a2a_agent_execution.RequestContext,
        event_queue: a2a_server.events.EventQueue,
    ) -> None:
        self._abort_controller.abort()

    async def _process_events(
        self, emitter: Emitter, context: a2a_agent_execution.RequestContext, updater: a2a_server_tasks.TaskUpdater
    ) -> None:
        pass


class ReActAgentExecutor(BaseA2AAgentExecutor):
    @override
    async def _process_events(
        self,
        emitter: Emitter,
        context: a2a_agent_execution.RequestContext,
        updater: a2a_server_tasks.TaskUpdater,
    ) -> None:
        async def process_event(data: ReActAgentUpdateEvent, event: EventMeta) -> None:
            text = ""
            if isinstance(data.data, ReActAgentIterationResult):
                if data.data.final_answer:
                    text = data.data.final_answer
                elif data.data.tool_output:
                    text = data.data.tool_output
                elif data.data.tool_name or data.data.tool_input:
                    text = to_json({"tool_name": data.data.tool_name, "tool_input": data.data.tool_input})
                elif data.data.thought:
                    text = data.data.thought

            await updater.start_work(
                a2a_utils.new_agent_text_message(
                    text,
                    context.context_id,
                    context.task_id,
                )
                if isinstance(data.data, ReActAgentIterationResult)
                else a2a_utils.new_agent_parts_message(parts=[a2a_types.Part(root=a2a_types.DataPart(data=data.data))]),
            )

        emitter.on("update", process_event)


class ToolCallingAgentExecutor(BaseA2AAgentExecutor):
    @override
    async def _process_events(
        self,
        emitter: Emitter,
        context: a2a_agent_execution.RequestContext,
        updater: a2a_server_tasks.TaskUpdater,
    ) -> None:
        last_msg = None

        async def process_event(
            data: RequirementAgentStartEvent
            | RequirementAgentSuccessEvent
            | ToolCallingAgentStartEvent
            | ToolCallingAgentSuccessEvent,
            event: EventMeta,
        ) -> None:
            nonlocal last_msg
            messages = data.state.memory.messages
            if last_msg is None:
                last_msg = messages[-1]

            cur_index = find_index(messages, lambda msg: msg is last_msg, fallback=-1, reverse_traversal=True)
            for message in messages[cur_index + 1 :]:
                if (isinstance(message, ToolMessage) and message.content[0].tool_name == "final_answer") or (
                    isinstance(message, AssistantMessage)
                    and isinstance(message.content[0], MessageToolCallContent)
                    and message.content[0].tool_name == "final_answer"
                ):
                    continue

                await updater.start_work(
                    a2a_utils.new_agent_parts_message(
                        parts=[
                            a2a_types.Part(
                                root=a2a_types.TextPart(text=content)
                                if isinstance(content, str)
                                else a2a_types.DataPart(data=content.model_dump())
                            )
                            for content in message.content
                        ]
                    ),
                )
                last_msg = message

        emitter.on("start", process_event)
        emitter.on("success", process_event)


def _extract_request_messages(context: a2a_agent_execution.RequestContext) -> list[AnyMessage]:
    return [
        convert_a2a_to_framework_message(message)
        for message in (
            context.current_task.history
            if context.current_task and context.current_task.history
            else [context.message]
            if context.message
            else []
        )
    ]
