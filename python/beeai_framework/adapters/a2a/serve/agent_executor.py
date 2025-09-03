# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing_extensions import TypeVar, override

from beeai_framework.adapters.a2a.agents._utils import convert_a2a_to_framework_message
from beeai_framework.agents.experimental.events import RequirementAgentSuccessEvent
from beeai_framework.serve import MemoryManager, init_agent_memory
from beeai_framework.utils.cancellation import AbortController
from beeai_framework.utils.cloneable import Cloneable

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
from beeai_framework.agents.tool_calling.events import (
    ToolCallingAgentSuccessEvent,
)
from beeai_framework.backend.message import (
    AnyMessage,
)
from beeai_framework.logger import Logger

AnyAgentLike = TypeVar("AnyAgentLike", bound=AnyAgent, default=AnyAgent)

logger = Logger(__name__)


class BaseA2AAgentExecutor(a2a_agent_execution.AgentExecutor):
    def __init__(self, agent: AnyAgentLike, agent_card: a2a_types.AgentCard, *, memory_manager: MemoryManager) -> None:
        super().__init__()
        self._agent = agent
        self.agent_card = agent_card
        self._abort_controller = AbortController()
        self._memory_manager = memory_manager

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
            response = await cloned_agent.run(new_messages, signal=self._abort_controller.signal)

            await updater.complete(
                a2a_utils.new_agent_text_message(
                    response.last_message.text,
                    context.context_id,
                    context.task_id,
                )
            )

        except Exception as e:
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


class TollCallingAgentExecutor(BaseA2AAgentExecutor):
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

        last_msg: AnyMessage | None = None
        try:
            async for data, _ in cloned_agent.run(new_messages, signal=self._abort_controller.signal):
                messages = data.state.memory.messages
                if last_msg is None:
                    last_msg = messages[-1]

                if isinstance(data, ToolCallingAgentSuccessEvent) and data.state.result is not None:
                    await updater.complete(
                        a2a_utils.new_agent_text_message(
                            data.state.result.text,
                            context.context_id,
                            context.task_id,
                        )
                    )
                if isinstance(data, RequirementAgentSuccessEvent) and data.state.answer is not None:
                    await updater.complete(
                        a2a_utils.new_agent_text_message(
                            data.state.answer.text,
                            context.context_id,
                            context.task_id,
                        )
                    )

        except Exception as e:
            await updater.failed(
                message=a2a_utils.new_agent_text_message(str(e)),
            )


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
