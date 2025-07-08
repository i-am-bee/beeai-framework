# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing_extensions import TypeVar, override

from beeai_framework.adapters.a2a.agents._utils import convert_a2a_to_framework_message
from beeai_framework.agents.errors import AgentError
from beeai_framework.agents.experimental.events import RequirementAgentSuccessEvent
from beeai_framework.utils.cancellation import AbortController

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
from beeai_framework.utils.lists import find_index

AnyAgentLike = TypeVar("AnyAgentLike", bound=AnyAgent, default=AnyAgent)


class BaseA2AAgentExecutor(a2a_agent_execution.AgentExecutor):
    def __init__(self, agent: AnyAgentLike, agent_card: a2a_types.AgentCard) -> None:
        super().__init__()
        self._agent = agent
        self.agent_card = agent_card
        self._abort_controller = AbortController()

    @override
    async def execute(
        self,
        context: a2a_agent_execution.RequestContext,
        event_queue: a2a_server.events.EventQueue,
    ) -> None:
        if not context.message:
            raise AgentError("No message provided")

        updater = a2a_server_tasks.TaskUpdater(event_queue, context.task_id, context.context_id)  # type: ignore[arg-type]
        if not context.current_task:
            context.current_task = a2a_utils.new_task(context.message)
            await updater.submit()
        assert context.current_task is not None

        self._agent.memory.reset()
        await self._agent.memory.add_many(
            [convert_a2a_to_framework_message(message) for message in context.current_task.history or []]
        )

        await updater.start_work()
        try:
            response = await self._agent.run(signal=self._abort_controller.signal)

            await updater.complete(
                a2a_utils.new_agent_text_message(
                    response.result.text,
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
        if not context.message:
            raise AgentError("No message provided")

        updater = a2a_server_tasks.TaskUpdater(event_queue, context.task_id, context.context_id)  # type: ignore[arg-type]
        if not context.current_task:
            context.current_task = a2a_utils.new_task(context.message)
            await updater.submit()
        assert context.current_task is not None

        self._agent.memory.reset()
        await self._agent.memory.add_many(
            [
                convert_a2a_to_framework_message(message)
                for message in (context.current_task.history or [])
                if all(isinstance(part.root, a2a_types.TextPart) for part in message.parts)
            ]
        )

        await updater.start_work()

        last_msg: AnyMessage | None = None
        try:
            async for data, _ in self._agent.run(signal=self._abort_controller.signal):
                messages = data.state.memory.messages
                if last_msg is None:
                    last_msg = messages[-1]

                cur_index = find_index(messages, lambda msg: msg is last_msg, fallback=-1, reverse_traversal=True)  # noqa: B023
                for message in messages[cur_index + 1 :]:
                    await updater.update_status(
                        a2a_types.TaskState.working,
                        message=a2a_utils.new_agent_parts_message(
                            parts=[
                                a2a_types.Part(root=a2a_types.DataPart(data=content.model_dump()))
                                for content in message.content
                            ]
                        ),
                    )
                    last_msg = message

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
