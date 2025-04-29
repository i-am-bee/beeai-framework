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

from functools import reduce

try:
    from acp_sdk.client import Client
    from acp_sdk.models import Message, MessagePart, RunCompletedEvent, RunFailedEvent
    from acp_sdk.models.errors import Error
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [acp] not found.\nRun 'pip install beeai-framework[acp]' to install."
    ) from e

from beeai_framework.agents.base import BaseAgent
from beeai_framework.agents.errors import AgentError
from beeai_framework.agents.experimental.remote.events import (
    RemoteAgentErrorEvent,
    RemoteAgentUpdateEvent,
    remote_agent_event_types,
)
from beeai_framework.agents.experimental.remote.types import (
    RemoteAgentInput,
    RemoteAgentRunOutput,
)
from beeai_framework.backend.message import AnyMessage, AssistantMessage, UserMessage
from beeai_framework.backend.message import Message as BeeAIMessage
from beeai_framework.context import Run, RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.memory import BaseMemory
from beeai_framework.utils import AbortSignal


class RemoteAgent(BaseAgent[RemoteAgentRunOutput]):
    def __init__(self, agent_name: str, *, url: str, memory: BaseMemory) -> None:
        super().__init__()
        self._memory = memory
        self.input = RemoteAgentInput(agent_name=agent_name, url=url)

    def run(
        self,
        input: str | AnyMessage | Message | list[str] | list[AnyMessage] | list[Message],
        *,
        signal: AbortSignal | None = None,
    ) -> Run[RemoteAgentRunOutput]:
        async def handler(context: RunContext) -> RemoteAgentRunOutput:
            async with Client(base_url=self.input.url) as client:
                inputs = (
                    [self._convert_to_acp_message(i) for i in input]
                    if isinstance(input, list)
                    else [self._convert_to_acp_message(input)]
                )

                last_event = None
                async for event in client.run_stream(agent=self.input.agent_name, input=inputs):
                    last_event = event
                    envet_dict = event.model_dump(exclude={"type"})
                    await context.emitter.emit("update", RemoteAgentUpdateEvent(key=event.type, value=envet_dict))

                if last_event is None:
                    raise AgentError("No event received from agent.")

                if isinstance(last_event, RunFailedEvent):
                    message = (
                        last_event.run.error.message
                        if isinstance(last_event.run.error, Error)
                        else "Something went wrong with the agent communication."
                    )
                    await context.emitter.emit(
                        "error",
                        RemoteAgentErrorEvent(message=message),
                    )
                    raise AgentError(message)
                elif isinstance(last_event, RunCompletedEvent):
                    response = str(reduce(lambda x, y: x + y, last_event.run.output))

                    input_messages = (
                        [self._convert_to_message(i) for i in input]
                        if isinstance(input, list)
                        else [self._convert_to_message(input)]
                    )
                    assistant_message = AssistantMessage(response, meta={"event": last_event})
                    await self.memory.add_many(input_messages)
                    await self.memory.add(assistant_message)

                    return RemoteAgentRunOutput(result=assistant_message, event=last_event)
                else:
                    return RemoteAgentRunOutput(result=AssistantMessage("No response from agent."), event=last_event)

        return self._to_run(
            handler,
            signal=signal,
            run_params={
                "prompt": input,
                "signal": signal,
            },
        )

    async def check_agent_exists(
        self,
    ) -> None:
        try:
            async with Client(base_url=self.input.url) as client:
                agents = [agent async for agent in client.agents()]
                agent = any(agent.name == self.input.agent_name for agent in agents)
                if not agent:
                    raise AgentError(f"Agent {self.input.agent_name} does not exist.")
        except Exception as e:
            raise AgentError("Can't connect to ACP agent.", cause=e)

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["agent", "remote"],
            creator=self,
            events=remote_agent_event_types,
        )

    @property
    def memory(self) -> BaseMemory:
        return self._memory

    @memory.setter
    def memory(self, memory: BaseMemory) -> None:
        self._memory = memory

    async def clone(self) -> "RemoteAgent":
        cloned = RemoteAgent(self.input.agent_name, url=self.input.url, memory=self.memory)
        cloned.emitter = await self.emitter.clone()
        return cloned

    def _convert_to_message(self, input: str | AnyMessage | Message) -> AnyMessage:
        if isinstance(input, str):
            return UserMessage(input)
        elif isinstance(input, BeeAIMessage):
            return input
        elif isinstance(input, Message):
            return UserMessage(str(input))
        else:
            raise ValueError("Unsupported input type")

    def _convert_to_acp_message(self, input: str | AnyMessage | Message) -> Message:
        if isinstance(input, str):
            return Message(parts=[MessagePart(content=input, role="user")])  # type: ignore[call-arg]
        elif isinstance(input, BeeAIMessage):
            return Message(parts=[MessagePart(content=input.text, role=input.role)])  # type: ignore[call-arg]
        elif isinstance(input, Message):
            return input
        else:
            raise ValueError("Unsupported input type")
