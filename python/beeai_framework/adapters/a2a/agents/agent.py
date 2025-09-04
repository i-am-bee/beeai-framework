# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import Unpack
from uuid import uuid4

import httpx
from a2a.types import TransportProtocol

from beeai_framework.adapters.a2a.agents._utils import convert_a2a_to_framework_message
from beeai_framework.backend import UserMessage
from beeai_framework.utils.strings import to_safe_word

try:
    import a2a.client as a2a_client
    import a2a.types as a2a_types
    import grpc

    from beeai_framework.adapters.a2a.agents.events import (
        A2AAgentErrorEvent,
        A2AAgentUpdateEvent,
        a2a_agent_event_types,
    )
    from beeai_framework.adapters.a2a.agents.types import (
        A2AAgentOutput,
    )
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
        agent_card_url: str | None = None,
        agent_card: a2a_types.AgentCard | None = None,
        memory: BaseMemory,
        grpc_client_credentials: grpc.ChannelCredentials | None = None,
    ) -> None:
        super().__init__()
        self._agent_card_url = agent_card_url
        if agent_card:
            self._name: str = agent_card.name
            self._url: str = agent_card.url
        elif url:
            self._name = f"agent_{url.split(':')[-1]}"
            self._url = url
        else:
            raise ValueError("Either url or agent_card must be provided.")
        if not memory.is_empty():
            raise ValueError("Memory must be empty before setting.")
        self._memory: BaseMemory = memory
        self._agent_card: a2a_types.AgentCard | None = agent_card
        self._context_id: str | None = None
        self._task_id: str | None = None
        self._reference_task_ids: list[str] = []
        self._grpc_client_credentials = grpc_client_credentials

    @property
    def name(self) -> str:
        return self._name

    @runnable_entry
    async def run(
        self, input: str | list[AnyMessage] | AnyMessage | a2a_types.Message, /, **kwargs: Unpack[A2AAgentOptions]
    ) -> A2AAgentOutput:
        self._context_id = kwargs.get("context_id") or self._context_id
        self._task_id = kwargs.get("task_id")
        if kwargs.get("clear_context"):
            self._context_id = None
            self._reference_task_ids.clear()

        context = RunContext.get()
        async with httpx.AsyncClient() as httpx_client:
            card_resolver = a2a_client.A2ACardResolver(httpx_client, self._agent_card_url or self._url)

            # get agent card
            try:
                self._agent_card = await card_resolver.get_agent_card()
            except Exception as e:
                raise RuntimeError("Failed to fetch the public agent card. Cannot continue.") from e

            if self._agent_card.supports_authenticated_extended_card:
                logger.warning("\nPublic card supports authenticated extended card but this is not supported yet.")

            if not self._agent_card:
                card_resolver = a2a_client.A2ACardResolver(httpx_client, self._agent_card_url or self._url)
                self._agent_card = await card_resolver.get_agent_card()

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
                        TransportProtocol.jsonrpc,
                        TransportProtocol.grpc,
                        TransportProtocol.http_json,
                    ],
                )
            ).create(self._agent_card)

            last_event: a2a_client.ClientEvent | a2a_types.Message | None = None
            messages: list[a2a_types.Message] = []
            task: a2a_types.Task | None = None

            # send request
            try:
                async for event in client.send_message(self._convert_to_a2a_message(input)):
                    last_event = event

                    if isinstance(event, a2a_types.Message):
                        messages.append(event)

                    elif isinstance(event, tuple):
                        task = event[0]

                    await context.emitter.emit("update", A2AAgentUpdateEvent(value=event))

                # check if we received any event
                if last_event is None:
                    raise AgentError("No result received from agent.")

                # insert input into memory
                input_messages = _convert_messages_to_framework_messages(input)
                await self.memory.add_many(input_messages)

                if task:
                    if task.status.state is not a2a_types.TaskState.completed:
                        logger.warning(f"Task status ({task.status.state}) is not complete.")

                    self._context_id = task.context_id
                    self._task_id = task.id
                    if self._task_id and self._task_id not in self._reference_task_ids:
                        self._reference_task_ids.append(self._task_id)

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
                    err.message if hasattr(err, "message") else err.error if hasattr(err, "error") else "Unknown error"
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

    async def check_agent_exists(self) -> None:
        try:
            async with httpx.AsyncClient() as httpx_client:
                card_resolver = a2a_client.A2ACardResolver(httpx_client, self._agent_card_url or self._url)
                agent_card = await card_resolver.get_agent_card()
                if not agent_card:
                    raise AgentError(f"Agent {self._name} does not exist.")
        except Exception as e:
            raise AgentError("Can't connect to ACP agent.", cause=e)

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["a2a", "agent", to_safe_word(self._name)],
            creator=self,
            events=a2a_agent_event_types,
        )

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
            agent_card_url=self._agent_card_url,
            agent_card=self._agent_card,
            memory=await self.memory.clone(),
        )
        cloned.emitter = await self.emitter.clone()
        return cloned

    def _convert_to_a2a_message(
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
            )
        elif isinstance(input, list) and input and isinstance(input[-1], Message):
            return a2a_types.Message(
                role=a2a_types.Role.agent if input[-1].role == Role.ASSISTANT else a2a_types.Role.user,
                parts=[a2a_types.Part(root=a2a_types.TextPart(text=input[-1].text))],
                message_id=uuid4().hex,
                context_id=self._context_id,
                task_id=self._task_id,
                reference_task_ids=self._reference_task_ids,
            )
        elif isinstance(input, a2a_types.Message):
            return input
        else:
            raise ValueError("Unsupported input type")


def _convert_messages_to_framework_messages(
    input: str | list[AnyMessage] | AnyMessage | a2a_types.Message | a2a_types.Artifact,
) -> list[AnyMessage]:
    return input if isinstance(input, list) else [_convert_message_to_framework_message(input)]


def _convert_message_to_framework_message(
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
