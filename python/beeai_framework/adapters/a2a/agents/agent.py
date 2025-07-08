# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncGenerator
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
        A2AAgentRunOutput,
    )
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [a2a] not found.\nRun 'pip install \"beeai-framework[a2a]\"' to install."
    ) from e

from beeai_framework.agents.base import BaseAgent
from beeai_framework.agents.errors import AgentError
from beeai_framework.backend.message import (
    AnyMessage,
    AssistantMessage,
    Message,
    Role,
    UserMessage,
)
from beeai_framework.context import Run, RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.logger import Logger
from beeai_framework.memory import BaseMemory
from beeai_framework.utils import AbortSignal

logger = Logger(__name__)


class A2AAgent(BaseAgent[A2AAgentRunOutput]):
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

    @property
    def name(self) -> str:
        return self._name

    def run(
        self,
        input: str | AnyMessage | a2a_types.Message,
        *,
        context_id: str | None = None,
        task_id: str | None = None,
        signal: AbortSignal | None = None,
    ) -> Run[A2AAgentRunOutput]:
        self._context_id = context_id or self._context_id
        self._task_id = task_id or self._task_id

        async def handler(context: RunContext) -> A2AAgentRunOutput:
            async with httpx.AsyncClient() as httpx_client:
                client: a2a_client.A2AClient = await a2a_client.A2AClient.get_client_from_agent_card_url(
                    httpx_client, self._url
                )

                streaming_request: a2a_types.SendStreamingMessageRequest = a2a_types.SendStreamingMessageRequest(
                    id=str(uuid4().hex), params=a2a_types.MessageSendParams(message=self._convert_to_a2a_message(input))
                )

                last_event_with_data = None
                stream_response: AsyncGenerator[a2a_types.SendStreamingMessageResponse] = client.send_message_streaming(
                    streaming_request
                )
                async for event in stream_response:
                    if last_event_with_data is None or has_content(event):
                        last_event_with_data = event
                    # emit all events as updates
                    await context.emitter.emit(
                        "update", A2AAgentUpdateEvent(value=event.model_dump(mode="json", exclude_none=True))
                    )

                # check if we received a any response
                if last_event_with_data is None:
                    raise AgentError("No result received from agent.")

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
                    self._context_id = response.contextId
                    self._task_id = response.id if isinstance(response, a2a_types.Task) else response.taskId

                    # add input message to memory
                    input_message: AnyMessage = self._convert_message_to_framework_message(input)
                    await self.memory.add(input_message)

                    # retrieve the assistant's response
                    assistant_message = None
                    if isinstance(response, a2a_types.Message):
                        assistant_message = self._convert_message_to_framework_message(response)
                    elif isinstance(response, a2a_types.TaskArtifactUpdateEvent):
                        if not response.lastChunk:
                            raise AgentError(
                                "Agent's response is not complete.", context=last_event_with_data.model_dump()
                            )

                        assistant_message = self._convert_message_to_framework_message(response.artifact)
                    elif isinstance(response, a2a_types.Task | a2a_types.TaskStatusUpdateEvent):
                        if isinstance(response, a2a_types.TaskStatusUpdateEvent) and not response.final:
                            logger.warning("Agent's task update event is not final.")
                        if response.status.state != a2a_types.TaskState.completed:
                            logger.warning("Agent's task is not completed.")
                        if not response.status.message:
                            if (
                                isinstance(response, a2a_types.Task)
                                and response.artifacts
                                and len(response.artifacts) > 0
                            ):
                                assistant_message = self._convert_message_to_framework_message(response.artifacts[-1])
                            else:
                                return A2AAgentRunOutput(
                                    result=AssistantMessage("No response from agent."), event=last_event_with_data
                                )
                        else:
                            assistant_message = self._convert_message_to_framework_message(response.status.message)
                    else:
                        raise AgentError("Invalid response from agent.", context=last_event_with_data.model_dump())

                    # add assistant message to memory
                    await self.memory.add(assistant_message)
                    return A2AAgentRunOutput(result=assistant_message, event=last_event_with_data)
                else:
                    return A2AAgentRunOutput(
                        result=AssistantMessage("No response from agent."), event=last_event_with_data
                    )

        return self._to_run(
            handler,
            signal=signal,
            run_params={
                "prompt": input,
                "signal": signal,
            },
        )

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

    def _convert_to_a2a_message(self, input: str | AnyMessage | a2a_types.Message) -> a2a_types.Message:
        if isinstance(input, str):
            return a2a_types.Message(
                role=a2a_types.Role.user,
                parts=[a2a_types.Part(root=a2a_types.TextPart(text=input))],
                messageId=uuid4().hex,
                contextId=self._context_id,
                taskId=self._task_id,
            )
        elif isinstance(input, Message):
            return a2a_types.Message(
                role=a2a_types.Role.agent if input.role == Role.ASSISTANT else a2a_types.Role.user,
                parts=[a2a_types.Part(root=a2a_types.TextPart(text=input.text))],
                messageId=uuid4().hex,
                contextId=self._context_id,
                taskId=self._task_id,
            )
        elif isinstance(input, a2a_types.Message):
            return input
        else:
            raise ValueError("Unsupported input type")
