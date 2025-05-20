# Copyright 2025 © BeeAI a Series of LF Projects, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import AsyncGenerator
from functools import reduce
from uuid import uuid4

import httpx

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
from beeai_framework.memory import BaseMemory
from beeai_framework.utils import AbortSignal


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
        self._memory = memory
        self._agent_card = agent_card

    @property
    def name(self) -> str:
        return self._name

    def run(
        self,
        input: str | AnyMessage | a2a_types.Message,
        *,
        signal: AbortSignal | None = None,
    ) -> Run[A2AAgentRunOutput]:
        async def handler(context: RunContext) -> A2AAgentRunOutput:
            async with httpx.AsyncClient() as httpx_client:
                client: a2a_client.A2AClient = await a2a_client.A2AClient.get_client_from_agent_card_url(
                    httpx_client, self._url
                )

                streaming_request: a2a_types.SendStreamingMessageRequest = a2a_types.SendStreamingMessageRequest(
                    params=a2a_types.MessageSendParams(message=self._convert_to_a2a_message(input))
                )

                last_event = None
                stream_response: AsyncGenerator[a2a_types.SendStreamingMessageResponse] = client.send_message_streaming(
                    streaming_request
                )
                async for event in stream_response:
                    last_event = event
                    await context.emitter.emit(
                        "update", A2AAgentUpdateEvent(value=event.model_dump(mode="json", exclude_none=True))
                    )

                if last_event is None:
                    raise AgentError("No event received from agent.")

                if isinstance(last_event, a2a_types.JSONRPCErrorResponse):
                    await context.emitter.emit(
                        "error",
                        A2AAgentErrorEvent(message=last_event.root.error.message),
                    )
                    raise AgentError(last_event.root.error.message)
                elif isinstance(last_event, a2a_types.SendStreamingMessageSuccessResponse):
                    response = last_event.root.result
                    if not isinstance(response, a2a_types.Message):
                        raise AgentError("Invalid response from agent.")
                    input_message: AnyMessage = self._convert_to_framework_message(input)
                    assistant_message: AnyMessage = self._convert_to_framework_message(response)

                    await self.memory.add(input_message)
                    await self.memory.add(assistant_message)

                    return A2AAgentRunOutput(result=assistant_message, event=last_event)
                else:
                    return A2AAgentRunOutput(result=AssistantMessage("No response from agent."), event=last_event)

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
        self._memory = memory

    async def clone(self) -> "A2AAgent":
        cloned = A2AAgent(url=self._url, agent_card=self._agent_card, memory=await self.memory.clone())
        cloned.emitter = await self.emitter.clone()
        return cloned

    def _convert_to_framework_message(self, input: str | AnyMessage | a2a_types.Message) -> AnyMessage:
        if isinstance(input, str):
            return UserMessage(input)
        elif isinstance(input, Message):
            return input
        elif isinstance(input, a2a_types.Message):
            if all(isinstance(part.root, a2a_types.TextPart) for part in input.parts):
                return AssistantMessage(
                    str(reduce(lambda x, y: x + y, input.parts).root.text),  # type: ignore
                    meta={"event": input},
                )
            else:
                raise ValueError("Unsupported response part type")
        else:
            raise ValueError(f"Unsupported input type {type(input)}")

    def _convert_to_a2a_message(self, input: str | AnyMessage | a2a_types.Message) -> a2a_types.Message:
        if isinstance(input, str):
            return a2a_types.Message(
                role=a2a_types.Role.user,
                parts=[a2a_types.Part(root=a2a_types.TextPart(text=input))],
                messageId=uuid4().hex,
            )
        elif isinstance(input, Message):
            return a2a_types.Message(
                role=a2a_types.Role.agent if input.role == Role.ASSISTANT else a2a_types.Role.user,
                parts=[a2a_types.Part(root=a2a_types.TextPart(text=input.text))],
                messageId=uuid4().hex,
            )
        elif isinstance(input, a2a_types.Message):
            return input
        else:
            raise ValueError("Unsupported input type")
