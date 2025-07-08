# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from functools import reduce

from beeai_framework.utils.strings import to_safe_word

try:
    import acp_sdk.client as acp_client
    import acp_sdk.models as acp_models

    from beeai_framework.adapters.acp.agents.events import (
        ACPAgentErrorEvent,
        ACPAgentUpdateEvent,
        acp_agent_event_types,
    )
    from beeai_framework.adapters.acp.agents.types import (
        ACPAgentRunOutput,
    )
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [acp] not found.\nRun 'pip install \"beeai-framework[acp]\"' to install."
    ) from e

from beeai_framework.agents.base import BaseAgent
from beeai_framework.agents.errors import AgentError
from beeai_framework.backend.message import AnyMessage, AssistantMessage, Message, UserMessage
from beeai_framework.context import Run, RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.memory import BaseMemory
from beeai_framework.utils import AbortSignal


class ACPAgent(BaseAgent[ACPAgentRunOutput]):
    def __init__(
        self, agent_name: str, *, url: str, memory: BaseMemory, session: acp_models.Session | None = None
    ) -> None:
        super().__init__()
        self._memory = memory
        self._url = url
        self._name = agent_name
        self._session = session or acp_models.Session()

    @property
    def name(self) -> str:
        return self._name

    def run(
        self,
        input: str | AnyMessage | acp_models.Message | list[str] | list[AnyMessage] | list[acp_models.Message],
        *,
        signal: AbortSignal | None = None,
    ) -> Run[ACPAgentRunOutput]:
        async def handler(context: RunContext) -> ACPAgentRunOutput:
            async with (
                acp_client.Client(base_url=self._url, session=self._session) as client,
            ):
                inputs = (
                    [self._convert_to_platform_message(i) for i in input]
                    if isinstance(input, list)
                    else [self._convert_to_platform_message(input)]
                )

                last_event = None
                async for event in client.run_stream(agent=self._name, input=inputs):
                    last_event = event
                    await context.emitter.emit(
                        "update",
                        ACPAgentUpdateEvent(key=event.type, value=event.model_dump(exclude={"type"})),
                    )

                if last_event is None:
                    raise AgentError("No event received from agent.")

                if isinstance(last_event, acp_models.RunFailedEvent):
                    message = (
                        last_event.run.error.message
                        if isinstance(last_event.run.error, acp_models.Error)
                        else "Something went wrong with the agent communication."
                    )
                    await context.emitter.emit(
                        "error",
                        ACPAgentErrorEvent(message=message),
                    )
                    raise AgentError(message)
                elif isinstance(last_event, acp_models.RunCompletedEvent):
                    response = str(reduce(lambda x, y: x + y, last_event.run.output))

                    input_messages = (
                        [self._convert_to_framework_message(i) for i in input]
                        if isinstance(input, list)
                        else [self._convert_to_framework_message(input)]
                    )

                    assistant_message = AssistantMessage(response, meta={"event": last_event})
                    await self.memory.add_many(input_messages)
                    await self.memory.add(assistant_message)

                    return ACPAgentRunOutput(result=assistant_message, event=last_event)
                else:
                    return ACPAgentRunOutput(result=AssistantMessage("No response from agent."), event=last_event)

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
            async with acp_client.Client(base_url=self._url) as client:
                agents = [agent async for agent in client.agents()]
                agent = any(agent.name == self._name for agent in agents)
                if not agent:
                    raise AgentError(f"Agent {self._name} does not exist.")
        except Exception as e:
            raise AgentError("Can't connect to ACP agent.", cause=e)

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["acp", "agent", to_safe_word(self._name)],
            creator=self,
            events=acp_agent_event_types,
        )

    @property
    def memory(self) -> BaseMemory:
        return self._memory

    @memory.setter
    def memory(self, memory: BaseMemory) -> None:
        self._memory = memory

    async def clone(self) -> "ACPAgent":
        cloned = ACPAgent(self._name, url=self._url, memory=await self.memory.clone())
        cloned.emitter = await self.emitter.clone()
        return cloned

    def _convert_to_framework_message(self, input: str | AnyMessage | acp_models.Message) -> AnyMessage:
        if isinstance(input, str):
            return UserMessage(input)
        elif isinstance(input, Message):
            return input
        elif isinstance(input, acp_models.Message):
            return UserMessage(str(input))
        else:
            raise ValueError("Unsupported input type")

    def _convert_to_platform_message(self, input: str | AnyMessage | acp_models.Message) -> acp_models.Message:
        if isinstance(input, str):
            return acp_models.Message(parts=[acp_models.MessagePart(content=input, role="user")])  # type: ignore[call-arg]
        elif isinstance(input, Message):
            return acp_models.Message(parts=[acp_models.MessagePart(content=input.text, role=input.role)])  # type: ignore[call-arg]
        elif isinstance(input, acp_models.Message):
            return input
        else:
            raise ValueError("Unsupported input type")
