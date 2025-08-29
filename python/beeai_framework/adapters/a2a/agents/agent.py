# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncGenerator
from typing import Unpack
from uuid import uuid4

import httpx

from beeai_framework.adapters.a2a.agents._utils import convert_a2a_to_framework_message, has_content
from beeai_framework.utils.strings import to_safe_word

try:
    import a2a.client as a2a_client
    import a2a.types as a2a_types

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
    UserMessage,
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
        self, *, url: str | None = None, agent_card: a2a_types.AgentCard | None = None, memory: BaseMemory
    ) -> None:
        super().__init__()
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
            card_resolver = a2a_client.A2ACardResolver(httpx_client, self._url)

            try:
                self._agent_card = await card_resolver.get_agent_card()
            except Exception as e:
                raise RuntimeError("Failed to fetch the public agent card. Cannot continue.") from e

            if self._agent_card.supports_authenticated_extended_card:
                logger.warning("\nPublic card supports authenticated extended card but this is not supported yet.")

            client: a2a_client.A2AClient = a2a_client.A2AClient(httpx_client, self._agent_card)

            if not self._agent_card:
                card_resolver = a2a_client.A2ACardResolver(httpx_client, self._url)
                self._agent_card = await card_resolver.get_agent_card()

            if self._agent_card.capabilities.streaming:
                return await self._handle_streaming_request(client=client, input=input, context=context)
            else:
                return await self._handle_request(client=client, input=input, context=context)

    async def check_agent_exists(self) -> None:
        try:
            async with httpx.AsyncClient() as httpx_client:
                card_resolver = a2a_client.A2ACardResolver(httpx_client, self._url)
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

    async def _handle_request(
        self,
        *,
        client: a2a_client.A2AClient,
        input: str | list[AnyMessage] | AnyMessage | a2a_types.Message,
        context: RunContext,
    ) -> A2AAgentOutput:
        request: a2a_types.SendMessageRequest = a2a_types.SendMessageRequest(
            id=str(uuid4().hex),
            params=a2a_types.MessageSendParams(message=self._convert_to_a2a_message(input)),
        )
        result: a2a_types.SendMessageResponse = await client.send_message(request)

        if isinstance(result.root, a2a_types.JSONRPCErrorResponse):
            await context.emitter.emit(
                "error",
                A2AAgentErrorEvent(message=result.root.error.message or "Unknown error"),
            )
            raise AgentError(
                result.root.error.message or "Unknown error",
                context=result.model_dump(),
            )

        # process success
        elif isinstance(result.root, a2a_types.SendMessageSuccessResponse):
            response = result.root.result
            await self._update_context_and_add_input_to_memory(response, input)

            # retrieve the assistant's response
            if isinstance(response, a2a_types.Message):
                assistant_messages = [self._convert_message_to_framework_message(response)]
            elif isinstance(response, a2a_types.Task):
                if not response.status.message:
                    if response.artifacts and len(response.artifacts) > 0:
                        assistant_messages = [
                            self._convert_message_to_framework_message(artifact) for artifact in response.artifacts
                        ]
                    else:
                        return A2AAgentOutput(output=[AssistantMessage("No response from agent.")], event=result)
                else:
                    assistant_messages = [self._convert_message_to_framework_message(response.status.message)]
            else:
                raise AgentError("Invalid response from agent.", context=result.model_dump())

            # add assistant message to memory
            await self.memory.add_many(assistant_messages)
            return A2AAgentOutput(output=assistant_messages, event=result)

    async def _handle_streaming_request(
        self,
        *,
        client: a2a_client.A2AClient,
        input: str | list[AnyMessage] | AnyMessage | a2a_types.Message,
        context: RunContext,
    ) -> A2AAgentOutput:
        streaming_request: a2a_types.SendStreamingMessageRequest = a2a_types.SendStreamingMessageRequest(
            id=str(uuid4().hex),
            params=a2a_types.MessageSendParams(message=self._convert_to_a2a_message(input)),
        )

        last_event_with_data = None
        last_event = None
        stream_response: AsyncGenerator[a2a_types.SendStreamingMessageResponse] = client.send_message_streaming(
            streaming_request
        )
        streamed_artifacts: list[a2a_types.Artifact] = []
        async for event in stream_response:
            last_event = event
            if last_event_with_data is None or has_content(event):
                last_event_with_data = event

            if isinstance(event.root, a2a_types.SendStreamingMessageSuccessResponse):
                response = event.root.result
                if isinstance(response, a2a_types.TaskArtifactUpdateEvent):
                    existing_artifacts = [
                        artifact
                        for artifact in streamed_artifacts
                        if artifact.artifact_id == response.artifact.artifact_id
                    ]
                    if any(existing_artifacts) and response.append:
                        existing_artifacts[-1].parts.extend(response.artifact.parts)
                    else:
                        streamed_artifacts.append(response.artifact)
            # emit all events as updates
            await context.emitter.emit(
                "update", A2AAgentUpdateEvent(value=event.model_dump(mode="json", exclude_none=True))
            )

        # check if we received any response
        if last_event is None or last_event_with_data is None:
            raise AgentError("No result received from agent.")

        if isinstance(last_event.root, a2a_types.SendStreamingMessageSuccessResponse) and isinstance(
            last_event.root.result, a2a_types.Task | a2a_types.TaskStatusUpdateEvent
        ):
            if isinstance(last_event.root.result, a2a_types.TaskStatusUpdateEvent) and not last_event.root.result.final:
                logger.warning("Agent's task update event is not final.")
            if last_event.root.result.status.state != a2a_types.TaskState.completed:
                logger.warning("Agent's task is not completed.")

        # process error
        if isinstance(last_event_with_data.root, a2a_types.JSONRPCErrorResponse):
            await context.emitter.emit(
                "error",
                A2AAgentErrorEvent(message=last_event_with_data.root.error.message or "Unknown error"),
            )
            raise AgentError(
                last_event_with_data.root.error.message or "Unknown error",
                context=last_event_with_data.model_dump(),
            )

        # process success
        elif isinstance(last_event_with_data.root, a2a_types.SendStreamingMessageSuccessResponse):
            response = last_event_with_data.root.result
            await self._update_context_and_add_input_to_memory(response, input)

            # retrieve the assistant's response
            if len(streamed_artifacts) > 0:
                if len(streamed_artifacts) > 1:
                    await self.memory.add_many(
                        self._convert_message_to_framework_message(artifact) for artifact in streamed_artifacts[:-1]
                    )
                assistant_message = self._convert_message_to_framework_message(streamed_artifacts[-1])
            elif isinstance(response, a2a_types.Message):
                assistant_message = self._convert_message_to_framework_message(response)
            elif isinstance(response, a2a_types.TaskArtifactUpdateEvent):
                if not response.last_chunk:
                    logger.warning("Agent's response is not complete.")

                assistant_message = self._convert_message_to_framework_message(response.artifact)
            elif isinstance(response, a2a_types.Task | a2a_types.TaskStatusUpdateEvent):
                if not response.status.message:
                    if isinstance(response, a2a_types.Task) and response.artifacts and len(response.artifacts) > 0:
                        assistant_message = self._convert_message_to_framework_message(response.artifacts[-1])
                    else:
                        return A2AAgentOutput(
                            output=[AssistantMessage("No response from agent.")], event=last_event_with_data
                        )
                else:
                    assistant_message = self._convert_message_to_framework_message(response.status.message)
            else:
                raise AgentError("Invalid response from agent.", context=last_event_with_data.model_dump())

            # add assistant message to memory
            await self.memory.add(assistant_message)
            return A2AAgentOutput(output=[assistant_message], event=last_event_with_data)
        else:
            return A2AAgentOutput(output=[AssistantMessage("No response from agent.")], event=last_event_with_data)

    async def _update_context_and_add_input_to_memory(
        self,
        response: a2a_types.Task
        | a2a_types.Message
        | a2a_types.TaskStatusUpdateEvent
        | a2a_types.TaskArtifactUpdateEvent,
        input: str | list[AnyMessage] | AnyMessage | a2a_types.Message,
    ) -> None:
        self._context_id = response.context_id
        self._task_id = response.id if isinstance(response, a2a_types.Task) else response.task_id
        if self._task_id and self._task_id not in self._reference_task_ids:
            self._reference_task_ids.append(self._task_id)

        messages = self._convert_messages_to_framework_messages(input)
        await self.memory.add_many(messages)

    @property
    def memory(self) -> BaseMemory:
        return self._memory

    @memory.setter
    def memory(self, memory: BaseMemory) -> None:
        if not memory.is_empty():
            raise ValueError("Memory must be empty before setting.")
        self._memory = memory

    async def clone(self) -> "A2AAgent":
        cloned = A2AAgent(url=self._url, agent_card=self._agent_card, memory=await self.memory.clone())
        cloned.emitter = await self.emitter.clone()
        return cloned

    def _convert_message_to_framework_message(
        self, input: str | AnyMessage | a2a_types.Message | a2a_types.Artifact
    ) -> AnyMessage:
        if isinstance(input, str):
            return UserMessage(input)
        elif isinstance(input, Message):
            return input
        elif isinstance(input, a2a_types.Message | a2a_types.Artifact):
            return convert_a2a_to_framework_message(input)
        else:
            raise ValueError(f"Unsupported input type {type(input)}")

    def _convert_messages_to_framework_messages(
        self, input: str | list[AnyMessage] | AnyMessage | a2a_types.Message | a2a_types.Artifact
    ) -> list[AnyMessage]:
        return input if isinstance(input, list) else [self._convert_message_to_framework_message(input)]

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
