# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import contextlib
from collections.abc import Iterator
from typing import Generic

from typing_extensions import TypeVar, override

from beeai_framework.adapters.a2a.agents._utils import convert_a2a_to_framework_message
from beeai_framework.adapters.a2a.serve.context import A2AContext
from beeai_framework.emitter import Emitter
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.runnable import AnyRunnableTypeVar
from beeai_framework.serve import MemoryManager
from beeai_framework.utils.cancellation import AbortController, AbortSignal
from beeai_framework.utils.cloneable import Cloneable

try:
    import a2a.server.agent_execution as a2a_agent_execution
    import a2a.server.events as a2a_events
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

logger = Logger(__name__)


class BaseA2AExecutor(a2a_agent_execution.AgentExecutor, Generic[AnyRunnableTypeVar]):
    def __init__(
        self,
        runnable: AnyRunnableTypeVar,
        agent_card: a2a_types.AgentCard,
        *,
        memory_manager: MemoryManager,
    ) -> None:
        super().__init__()
        self._runnable = runnable
        self.agent_card = agent_card
        self._abort_controllers: dict[str, AbortController] = {}
        self._event_queues: dict[str, a2a_events.EventQueue] = {}
        self._memory_manager = memory_manager

    @override
    async def execute(
        self,
        context: a2a_agent_execution.RequestContext,
        event_queue: a2a_events.EventQueue,
    ) -> None:
        if not context.message:
            raise ValueError("No message found in the request context.")

        # registered before any await, so a cancel() that arrives during the setup still aborts this run
        with self._abort_signal(context.task_id, event_queue) as signal:
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

            # cancel() has already reported the canceled status for this task
            if signal.aborted:
                return

            try:
                with A2AContext(context=context, event_queue=event_queue):
                    # pyrefly: ignore [missing-attribute]
                    data = await cloned_runnable.run(messages, signal=signal)
                    # cancel() has already reported the canceled status for this task
                    if signal.aborted:
                        return
                    if memory is not None:
                        await memory.add(data.last_message)

                    await event_queue.enqueue_event(
                        a2a_utils.new_agent_text_message(
                            text=data.last_message.text, context_id=context.context_id, task_id=context.task_id
                        )
                    )

            except Exception as e:
                # cancel() has already reported the canceled status; any abort it didn't cause
                # (e.g. a timeout) leaves the signal untouched and is still reported
                if signal.aborted:
                    return
                logger.exception("Exception during execution")
                await event_queue.enqueue_event(a2a_utils.new_agent_text_message(str(e)))

    @override
    async def cancel(
        self,
        context: a2a_agent_execution.RequestContext,
        event_queue: a2a_events.EventQueue,
    ) -> None:
        if context.task_id and (controller := self._abort_controllers.get(context.task_id)):
            controller.abort()
        # a2a-sdk passes a queue tapped from the running task, its events never reach the streaming subscribers
        if context.task_id and context.task_id in self._event_queues:
            event_queue = self._event_queues[context.task_id]

        updater = a2a_server_tasks.TaskUpdater(event_queue, context.task_id, context.context_id)  # type: ignore[arg-type]
        await updater.cancel()

    @contextlib.contextmanager
    def _abort_signal(self, task_id: str | None, event_queue: a2a_events.EventQueue) -> Iterator[AbortSignal]:
        controller = AbortController()
        if task_id:
            self._abort_controllers[task_id] = controller
            self._event_queues[task_id] = event_queue
        try:
            yield controller.signal
        finally:
            # a follow-up request for the same task may have registered its own controller meanwhile
            if task_id and self._abort_controllers.get(task_id) is controller:
                del self._abort_controllers[task_id]
                del self._event_queues[task_id]

    async def _process_events(
        self, emitter: Emitter, context: a2a_agent_execution.RequestContext, updater: a2a_server_tasks.TaskUpdater
    ) -> None:
        pass
