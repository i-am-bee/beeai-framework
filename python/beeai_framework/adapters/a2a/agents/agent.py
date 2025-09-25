# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import Unpack
from uuid import uuid4

import httpx

from beeai_framework.adapters.a2a.agents._utils import convert_a2a_to_framework_message
from beeai_framework.adapters.a2a.agents.events import (
    A2AAgentErrorEvent,
    A2AAgentUpdateEvent,
    a2a_agent_event_types,
)
from beeai_framework.adapters.a2a.agents.types import (
    A2AAgentOutput,
)
from beeai_framework.backend import UserMessage
from beeai_framework.utils.strings import to_safe_word

try:
    import a2a.client as a2a_client
    import a2a.types as a2a_types
    import a2a.utils as a2a_utils
    import grpc
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [a2a] not found.\nRun 'pip install \"beeai-framework[a2a]\"' to install."
    ) from e

from beeai_framework.agents import AgentError, AgentOptions, BaseAgent
from beeai_framework.backend.message import (
    AnyMessage,
    AssistantMessage,
    Message,
    Role,
)
from beeai_framework.context import RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.logger import Logger
from beeai_framework.memory import BaseMemory
from beeai_framework.runnable import runnable_entry

logger = Logger(__name__)


class A2AAgentOptions(AgentOptions, total=False):
    context_id: str
    task_id: str
    clear_context: bool


class A2AAgent(BaseAgent[A2AAgentOutput]):
    def __init__(
        self,
        *,
        url: str | None = None,
        agent_card_path: str = a2a_utils.AGENT_CARD_WELL_KNOWN_PATH,
        agent_card: a2a_types.AgentCard | None = None,
        memory: BaseMemory,
        grpc_client_credentials: grpc.ChannelCredentials | None = None,
    ) -> None:
        super().__init__()
        if agent_card:
            self._agent_card: a2a_types.AgentCard | None = agent_card
            self._url = None
        elif url:
            self._url = url
            self._agent_card = None
        else:
            raise ValueError("Either url or agent_card must be provided.")
        if not memory.is_empty():
            raise ValueError("Memory must be empty before setting.")

        self._agent_card_path = agent_card_path
        self._memory: BaseMemory = memory
        self._context_id: str | None = None
        self._task_id: str | None = None
        self._reference_task_ids: list[str] = []
        self._grpc_client_credentials = grpc_client_credentials

    @property
    def name(self) -> str:
        return self._agent_card.name if self._agent_card else f"agent_{(self._url or '').split(':')[-1]}"

    @runnable_entry
    async def run(
        self, input: str | list[AnyMessage] | AnyMessage | a2a_types.Message, /, **kwargs: Unpack[A2AAgentOptions]
    ) -> A2AAgentOutput:
        self.set_run_params(
            context_id=kwargs.get("context_id"),
            task_id=kwargs.get("task_id"),
            clear_context=kwargs.get("clear_context"),
        )

        context = RunContext.get()

        if self._agent_card is None:
            await self._load_agent_card()

        assert self._agent_card is not None, "Agent card should not be empty after loading."

        async with httpx.AsyncClient() as httpx_client:
            # create client
            client: a2a_client.Client = a2a_client.ClientFactory(
                config=a2a_client.ClientConfig(
                    streaming=True,
                    polling=True,
                    httpx_client=httpx_client,
                    grpc_channel_factory=(
                        lambda url: grpc.aio.secure_channel(url, self._grpc_client_credentials)
                        if self._grpc_client_credentials
                        else grpc.aio.insecure_channel(url)
                    ),
                    supported_transports=[
                        a2a_types.TransportProtocol.jsonrpc,
                        a2a_types.TransportProtocol.grpc,
                        a2a_types.TransportProtocol.http_json,
                    ],
                )
            ).create(self._agent_card)

            if self._agent_card.supports_authenticated_extended_card:
                self._agent_card = await client.get_card()

            last_event: a2a_client.ClientEvent | a2a_types.Message | None = None
            messages: list[a2a_types.Message] = []
            task: a2a_types.Task | None = None

            # send request
            try:
                async for event in client.send_message(self.convert_to_a2a_message(input)):
                    last_event = event

                    [task_id, context_id] = (
                        [event.task_id, event.context_id]
                        if isinstance(event, a2a_types.Message)
                        else [event[0].id, event[0].context_id]
                    )

                    if task_id and task_id != self._task_id:
                        self._task_id = task_id
                    if context_id and context_id != self._context_id:
                        self._context_id = context_id
                    if task_id and task_id not in self._reference_task_ids:
                        self._reference_task_ids.append(task_id)

                    if isinstance(event, a2a_types.Message):
                        messages.append(event)

                    elif isinstance(event, tuple):
                        task, update_event = event

                        if (
                            update_event
                            and isinstance(update_event, a2a_types.TaskStatusUpdateEvent)
                            and update_event.final
                            and update_event.status.state is not a2a_types.TaskState.input_required
                        ):
                            self._task_id = None

                    await context.emitter.emit("update", A2AAgentUpdateEvent(value=event))

                # check if we received any event
                if last_event is None:
                    raise AgentError("No result received from agent.")

                # insert input into memory
                input_messages = [input] if not isinstance(input, list) else input
                await self.memory.add_many([_convert_to_framework_message(m) for m in input_messages])

                if task:
                    if task.status.state is not a2a_types.TaskState.completed:
                        logger.warning(f"Task status ({task.status.state}) is not complete.")

                    results: Sequence[a2a_types.Message | a2a_types.Artifact]
                    if task.artifacts:
                        results = task.artifacts
                    elif task.history:
                        results = task.history
                    elif task.status.message:
                        results = [task.status.message]
                    else:
                        results = []
                else:
                    results = messages if messages else []

                # retrieve the assistant's response
                if results:
                    assistant_messages = [convert_a2a_to_framework_message(result) for result in results]
                    await self.memory.add_many(assistant_messages)
                    return A2AAgentOutput(output=[assistant_messages[-1]], event=last_event)
                else:
                    return A2AAgentOutput(output=[AssistantMessage("No response from agent.")], event=last_event)

            except a2a_client.A2AClientError as err:
                message = (
                    err.message
                    if hasattr(err, "message")
                    else str(err.error)
                    if hasattr(err, "error")
                    else "Unknown error"
                )
                await context.emitter.emit("error", A2AAgentErrorEvent(message=message))
                error_context = None
                if isinstance(last_event, a2a_types.Message):
                    error_context = last_event.model_dump(mode="json", exclude_none=True)
                elif isinstance(last_event, tuple) and last_event[1]:
                    error_context = last_event[1].model_dump(mode="json", exclude_none=True)
                raise AgentError(
                    message,
                    context=error_context if isinstance(last_event, tuple) and last_event[1] else None,
                    cause=err,
                )

    def set_run_params(
        self, *, context_id: str | None, task_id: str | None, clear_context: bool | None = False
    ) -> None:
        self._context_id = context_id or self._context_id
        self._task_id = task_id
        if clear_context:
            self._context_id = None
            self._reference_task_ids.clear()

    async def _load_agent_card(self) -> None:
        if self._agent_card:
            return
        if not self._url:
            raise AgentError("No url provided.")
        try:
            async with httpx.AsyncClient() as httpx_client:
                card_resolver = a2a_client.A2ACardResolver(
                    httpx_client, self._url, agent_card_path=self._agent_card_path
                )
                self._agent_card = await card_resolver.get_agent_card()
        except Exception as e:
            raise AgentError("Can't load agent card.", cause=e)

    async def check_agent_exists(self) -> None:
        await self._load_agent_card()
        if not self._agent_card:
            raise AgentError(f"Agent {self.name} does not exist.")

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["a2a", "agent", to_safe_word(self.name)],
            creator=self,
            events=a2a_agent_event_types,
        )

    @property
    def agent_card(self) -> a2a_types.AgentCard | None:
        return self._agent_card

    @property
    def memory(self) -> BaseMemory:
        return self._memory

    @memory.setter
    def memory(self, memory: BaseMemory) -> None:
        if not memory.is_empty():
            raise ValueError("Memory must be empty before setting.")
        self._memory = memory

    async def clone(self) -> "A2AAgent":
        cloned = A2AAgent(
            url=self._url,
            agent_card_path=self._agent_card_path,
            agent_card=self._agent_card,
            memory=await self.memory.clone(),
            grpc_client_credentials=self._grpc_client_credentials,
        )
        cloned.emitter = await self.emitter.clone()
        return cloned

    def convert_to_a2a_message(
        self, input: str | list[AnyMessage] | AnyMessage | a2a_types.Message
    ) -> a2a_types.Message:
        if isinstance(input, str):
            return a2a_types.Message(
                role=a2a_types.Role.user,
                parts=[a2a_types.Part(root=a2a_types.TextPart(text=input))],
                message_id=uuid4().hex,
                context_id=self._context_id,
                task_id=self._task_id,
                reference_task_ids=self._reference_task_ids,
            )
        elif isinstance(input, Message):
            return a2a_types.Message(
                role=a2a_types.Role.agent if input.role == Role.ASSISTANT else a2a_types.Role.user,
                parts=[a2a_types.Part(root=a2a_types.TextPart(text=input.text))],
                message_id=uuid4().hex,
                context_id=self._context_id,
                task_id=self._task_id,
                reference_task_ids=self._reference_task_ids,
                metadata=input.meta or None,
            )
        elif isinstance(input, list) and input and isinstance(input[-1], Message):
            return self.convert_to_a2a_message(input[-1])
        elif isinstance(input, a2a_types.Message):
            return input
        else:
            raise ValueError("Unsupported input type")


def _convert_to_framework_message(
    input: str | AnyMessage | a2a_types.Message | a2a_types.Artifact,
) -> AnyMessage:
    if isinstance(input, str):
        return UserMessage(input)
    elif isinstance(input, Message):
        return input
    elif isinstance(input, a2a_types.Message | a2a_types.Artifact):
        return convert_a2a_to_framework_message(input)
    else:
        raise ValueError(f"Unsupported input type {type(input)}")
