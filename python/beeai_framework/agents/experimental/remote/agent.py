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
from beeai_framework.backend.message import AnyMessage, AssistantMessage
from beeai_framework.backend.message import Message as BeeAIMessage
from beeai_framework.context import Run, RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.memory import BaseMemory
from beeai_framework.utils import AbortSignal


class RemoteAgent(BaseAgent[RemoteAgentRunOutput]):
    def __init__(self, agent_name: str, *, url: str) -> None:
        super().__init__()
        self.input = RemoteAgentInput(agent_name=agent_name, url=url)

    def run(
        self,
        input: str | AnyMessage | Message | list[str] | list[AnyMessage] | list[Message],
        *,
        signal: AbortSignal | None = None,
    ) -> Run[RemoteAgentRunOutput]:
        async def handler(context: RunContext) -> RemoteAgentRunOutput:
            async with Client(base_url=self.input.url) as client:

                def convert_to_acp_message(input: str | AnyMessage | Message) -> Message:
                    if isinstance(input, str):
                        return Message(parts=[MessagePart(content=input)])
                    elif isinstance(input, BeeAIMessage):
                        return Message(parts=[MessagePart(content=input.text)])
                    elif isinstance(input, Message):
                        return input
                    else:
                        raise ValueError("Unsupported input type")

                inputs = (
                    [convert_to_acp_message(i) for i in input]
                    if isinstance(input, list)
                    else [convert_to_acp_message(input)]
                )

                last_event = None
                async for event in client.run_stream(agent=self.input.agent_name, inputs=inputs):
                    last_event = event
                    envet_dict = event.model_dump()
                    del envet_dict["type"]
                    await context.emitter.emit("update", RemoteAgentUpdateEvent(key=event.type, value=envet_dict))

                if isinstance(last_event, RunFailedEvent) and isinstance(last_event.run.error, Error):
                    await context.emitter.emit(
                        "error",
                        RemoteAgentErrorEvent(message=last_event.run.error.message),
                    )
                if isinstance(last_event, RunCompletedEvent):
                    response = str(reduce(lambda x, y: x + y, last_event.run.outputs))
                    return RemoteAgentRunOutput(result=AssistantMessage(response), native_response=last_event.run)
                else:
                    return RemoteAgentRunOutput(
                        result=AssistantMessage("No response from agent."),
                        native_response=last_event.run
                        if last_event is not None and hasattr(last_event, "run")
                        else None,
                    )

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
        raise NotImplementedError()

    @memory.setter
    def memory(self, memory: BaseMemory) -> None:
        raise NotImplementedError()

    async def clone(self) -> "RemoteAgent":
        cloned = RemoteAgent(self.input.agent_name, url=self.input.url)
        cloned.emitter = await self.emitter.clone()
        return cloned
