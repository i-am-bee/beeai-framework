# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import ssl
from collections.abc import Callable, Mapping, Sequence
from contextlib import AsyncExitStack
from typing import Any, Literal, Unpack

import httpx
import httpx._types as httpx_types
from httpx import AsyncBaseTransport
from httpx._auth import Auth  # noqa: F401
from httpx._client import ClientState
from httpx._config import Proxy, Timeout  # noqa: F401
from httpx._models import Cookies, Headers, Request  # noqa: F401
from httpx._urls import URL, QueryParams  # noqa: F401
from pydantic import BaseModel, ConfigDict

from beeai_framework.adapters.a2a.agents._utils import convert_a2a_to_framework_message, convert_to_a2a_message
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
    import a2a.utils.errors as a2a_errors
    import grpc
    from a2a.server.tasks.task_manager import append_artifact_to_task
    from google.protobuf.json_format import MessageToDict
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [a2a] not found.\nRun 'pip install \"beeai-framework[a2a]\"' to install."
    ) from e

from beeai_framework.agents import AgentError, AgentMeta, AgentOptions, BaseAgent
from beeai_framework.backend.message import (
    AnyMessage,
    AssistantMessage,
    Message,
)
from beeai_framework.context import RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.logger import Logger
from beeai_framework.memory import BaseMemory, UnconstrainedMemory
from beeai_framework.runnable import runnable_entry

logger = Logger(__name__)

_TERMINAL_TASK_STATES = (
    a2a_types.TaskState.TASK_STATE_COMPLETED,
    a2a_types.TaskState.TASK_STATE_FAILED,
    a2a_types.TaskState.TASK_STATE_CANCELED,
    a2a_types.TaskState.TASK_STATE_REJECTED,
)


class A2AAgentOptions(AgentOptions, total=False):
    context_id: str
    task_id: str
    clear_context: bool
    a2a_context: a2a_client.ClientCallContext
    # The HTTP client should already be opened
    httpx_client: httpx.AsyncClient
    streaming: bool


class HttpxAsyncClientParameters(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    auth: httpx_types.AuthTypes | None = None
    params: httpx_types.QueryParamTypes | None = None
    headers: httpx_types.HeaderTypes | None = None
    cookies: httpx_types.CookieTypes | None = None
    verify: ssl.SSLContext | str | bool = True
    cert: httpx_types.CertTypes | None = None
    http1: bool = True
    http2: bool = False
    proxy: httpx_types.ProxyTypes | None = None
    mounts: Mapping[str, AsyncBaseTransport | None] | None = None
    timeout: httpx_types.TimeoutTypes = httpx.Timeout(30.0, read=None)
    follow_redirects: bool = False
    limits: httpx.Limits = httpx._config.DEFAULT_LIMITS
    max_redirects: int = httpx._config.DEFAULT_MAX_REDIRECTS
    event_hooks: Mapping[str, list[httpx._client.EventHook]] | None = None
    base_url: httpx._urls.URL | str = ""
    transport: AsyncBaseTransport | None = None
    trust_env: bool = True
    default_encoding: str | Callable[[bytes], str] = "utf-8"


class A2AAgentParameters(BaseModel):
    httpx_async_client: HttpxAsyncClientParameters = HttpxAsyncClientParameters()
    extract_messages_from: Literal["task-artifacts", "task-history", "task-status", "messages"] | None = None


class A2AAgent(BaseAgent[A2AAgentOutput]):
    def __init__(
        self,
        *,
        url: str | None = None,
        agent_card_path: str = a2a_utils.AGENT_CARD_WELL_KNOWN_PATH,
        agent_card: a2a_types.AgentCard | None = None,
        memory: BaseMemory,
        grpc_client_credentials: grpc.ChannelCredentials | None = None,
        parameters: A2AAgentParameters | None = None,
    ) -> None:
        super().__init__()
        if agent_card:
            self._agent_card: a2a_types.AgentCard | None = agent_card
            self._url: str | None = None
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
        self._parameters = parameters or A2AAgentParameters()

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

        async with AsyncExitStack() as stack:
            if kwargs.get("httpx_client") is not None:
                httpx_client = kwargs.get("httpx_client")
                if httpx_client and httpx_client._state is ClientState.UNOPENED:
                    httpx_client = await stack.enter_async_context(httpx_client)
            else:
                new_client = httpx.AsyncClient(**self._parameters.httpx_async_client.model_dump())
                httpx_client = await stack.enter_async_context(new_client)

            # create a2a client
            client: a2a_client.Client = a2a_client.ClientFactory(
                config=a2a_client.ClientConfig(
                    streaming=kwargs.get("streaming", self._agent_card.capabilities.streaming),
                    polling=False,
                    httpx_client=httpx_client,
                    grpc_channel_factory=(
                        lambda url: grpc.aio.secure_channel(url, self._grpc_client_credentials)
                        if self._grpc_client_credentials
                        else grpc.aio.insecure_channel(url)
                    ),
                    supported_protocol_bindings=[
                        a2a_utils.TransportProtocol.JSONRPC,
                        a2a_utils.TransportProtocol.GRPC,
                        a2a_utils.TransportProtocol.HTTP_JSON,
                    ],
                )
            ).create(self._agent_card)

            a2a_context = kwargs.get("a2a_context")
            if self._agent_card.capabilities.extended_agent_card:
                self._agent_card = await client.get_extended_agent_card(
                    a2a_types.GetExtendedAgentCardRequest(), context=a2a_context
                )

            last_event: a2a_types.StreamResponse | None = None
            messages: list[a2a_types.Message] = []
            task: a2a_types.Task | None = None

            # send request
            try:
                request = a2a_types.SendMessageRequest(message=self.convert_to_a2a_message(input))
                async for event in client.send_message(request, context=a2a_context):
                    last_event = event

                    if event.HasField("message"):
                        task_id, context_id = event.message.task_id, event.message.context_id
                    else:
                        task = _merge_task_event(task, event)
                        task_id, context_id = task.id, task.context_id

                    if task_id and task_id != self._task_id:
                        self._task_id = task_id
                    if context_id and context_id != self._context_id:
                        self._context_id = context_id
                    if task_id and task_id not in self._reference_task_ids:
                        self._reference_task_ids.append(task_id)

                    if event.HasField("message"):
                        messages.append(event.message)

                    elif event.HasField("status_update") and event.status_update.status.state in _TERMINAL_TASK_STATES:
                        self._task_id = None

                        if event.status_update.status.HasField("message"):
                            messages.append(event.status_update.status.message)

                    await context.emitter.emit("update", A2AAgentUpdateEvent(value=event))

                # check if we received any event
                if last_event is None:
                    raise AgentError("No result received from agent.")

                if task and task.status.state in _TERMINAL_TASK_STATES:
                    self._task_id = None

                # insert input into memory
                input_messages = [input] if not isinstance(input, list) else input
                await self.memory.add_many([_convert_to_framework_message(m) for m in input_messages])

                results: Sequence[a2a_types.Message | a2a_types.Artifact] = []
                if self._parameters.extract_messages_from is None:
                    if task:
                        if task.status.state != a2a_types.TaskState.TASK_STATE_COMPLETED:
                            logger.warning(
                                f"Task status ({a2a_types.TaskState.Name(task.status.state)}) is not complete."
                            )

                        agent_history = [h for h in task.history if h.role == a2a_types.Role.ROLE_AGENT]
                        if task.artifacts:
                            results = list(task.artifacts)
                        elif agent_history:
                            results = agent_history
                        elif task.status.HasField("message"):
                            results = [task.status.message]
                        else:
                            results = []
                    else:
                        results = messages
                else:
                    match self._parameters.extract_messages_from:
                        case "task-artifacts":
                            if task:
                                results = list(task.artifacts)
                            else:
                                logger.error("Cannot extract artifact from task, because task is not provided.")
                        case "task-history":
                            if task:
                                results = list(task.history)
                            else:
                                logger.error("Cannot extract history from task, because task is not provided.")
                        case "task-status":
                            if task:
                                results = [task.status.message] if task.status.HasField("message") else []
                            else:
                                logger.error("Cannot extract status message from task, because task is not provided.")
                        case "messages":
                            results = messages

                # retrieve the assistant's response
                if results:
                    assistant_messages = [convert_a2a_to_framework_message(result) for result in results]
                    await self.memory.add_many(assistant_messages)
                    return A2AAgentOutput(output=[assistant_messages[-1]], event=last_event)
                else:
                    return A2AAgentOutput(output=[AssistantMessage("No response from agent.")], event=last_event)

            # a2a-sdk 1.x raises the protocol errors reported by the agent as A2AError, not A2AClientError
            except a2a_errors.A2AError as err:
                await context.emitter.emit("error", A2AAgentErrorEvent(message=err.message))
                payload = last_event.WhichOneof("payload") if last_event else None
                raise AgentError(
                    err.message,
                    context=MessageToDict(getattr(last_event, payload), preserving_proto_field_name=True)
                    if last_event and payload
                    else None,
                    cause=err,
                )

        return A2AAgentOutput(output=[AssistantMessage("No response from agent.")], event=last_event)

    def set_run_params(
        self, *, context_id: str | None, task_id: str | None, clear_context: bool | None = False
    ) -> None:
        self._context_id = context_id or self._context_id
        self._task_id = task_id or self._task_id
        if clear_context:
            self._context_id = None
            self._task_id = None
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

    @property
    def meta(self) -> AgentMeta:
        return AgentMeta(
            name=self.name,
            description=(self._agent_card.description or "") if self._agent_card else "",
            tools=[],
            extra_description=None,
        )

    async def clone(self) -> "A2AAgent":
        cloned = A2AAgent(
            url=self._url,
            agent_card_path=self._agent_card_path,
            agent_card=self._agent_card,
            memory=UnconstrainedMemory(),
            grpc_client_credentials=self._grpc_client_credentials,
            parameters=self._parameters.model_copy(),
        )
        # the constructor and the memory setter only accept an empty memory, the clone keeps the history
        cloned._memory = await self.memory.clone()
        cloned.emitter = await self.emitter.clone()
        return cloned

    def convert_to_a2a_message(
        self,
        input: str | list[AnyMessage] | AnyMessage | a2a_types.Message,
        *,
        context_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> a2a_types.Message:
        return convert_to_a2a_message(
            input,
            metadata=metadata,
            context_id=context_id
            or (input.context_id if isinstance(input, a2a_types.Message) else None)
            or self._context_id,
            task_id=self._task_id if (not context_id or context_id == self._context_id) else None,
            reference_task_ids=self._reference_task_ids if not context_id else None,
        )


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


def _merge_task_event(task: a2a_types.Task | None, event: a2a_types.StreamResponse) -> a2a_types.Task:
    # a2a-sdk 1.x streams the task events as they come, the task is rebuilt as its server-side TaskManager does
    if event.HasField("task"):
        merged = a2a_types.Task()
        merged.CopyFrom(event.task)
        return merged

    update = event.status_update if event.HasField("status_update") else event.artifact_update
    if task is None:
        task = a2a_types.Task(id=update.task_id, context_id=update.context_id)
    if event.HasField("status_update"):
        if task.status.HasField("message"):
            task.history.append(task.status.message)
        task.status.CopyFrom(event.status_update.status)
    else:
        append_artifact_to_task(task, event.artifact_update)
    return task
