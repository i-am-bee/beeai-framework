# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Generic

from typing_extensions import TypeVar, override

from beeai_framework.adapters.a2a.agents._utils import convert_a2a_to_framework_message
from beeai_framework.emitter import Emitter
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.runnable import Runnable
from beeai_framework.serve import MemoryManager
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
from beeai_framework.logger import Logger

AnyAgentLike = TypeVar("AnyAgentLike", bound=AnyAgent, default=AnyAgent)
AnyRunnable = TypeVar("AnyRunnable", bound=Runnable[Any], default=Runnable[Any])

logger = Logger(__name__)


class BaseA2AExecutor(a2a_agent_execution.AgentExecutor, Generic[AnyRunnable]):
    def __init__(
        self,
        runnable: AnyRunnable,
        agent_card: a2a_types.AgentCard,
        *,
        memory_manager: MemoryManager,
    ) -> None:
        super().__init__()
        self._runnable = runnable
        self.agent_card = agent_card
        self._abort_controller = AbortController()
        self._memory_manager = memory_manager

    @override
    async def execute(
        self,
        context: a2a_agent_execution.RequestContext,
        event_queue: a2a_server.events.EventQueue,
    ) -> None:
        if not context.message:
            raise ValueError("No message found in the request context.")

        cloned_runnable = await self._runnable.clone() if isinstance(self._runnable, Cloneable) else self._runnable
        memory = None
        if context.context_id:
            try:
                memory = await self._memory_manager.get(context.context_id)
            except KeyError:
                memory = UnconstrainedMemory()
                await self._memory_manager.set(context.context_id, memory)

            await memory.add(convert_a2a_to_framework_message(context.message))

        messages = memory.messages if memory else [convert_a2a_to_framework_message(context.message)]

        try:
            data = await cloned_runnable.run(messages, signal=self._abort_controller.signal)
            if memory is not None:
                await memory.add(data.last_message)

            await event_queue.enqueue_event(
                a2a_utils.new_agent_text_message(
                    text=data.last_message.text, context_id=context.context_id, task_id=context.task_id
                )
            )

        except Exception as e:
            logger.exception("Exception during execution")
            await event_queue.enqueue_event(a2a_utils.new_agent_text_message(str(e)))

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
