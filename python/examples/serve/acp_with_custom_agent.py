import sys
import traceback
from collections.abc import AsyncGenerator

import acp_sdk.models as acp_models
import acp_sdk.server.context as acp_context
import acp_sdk.server.types as acp_types
from pydantic import BaseModel, InstanceOf

from beeai_framework.adapters.acp import AcpAgentServer, to_framework_message
from beeai_framework.adapters.acp.serve._agent import AcpAgent
from beeai_framework.agents.base import BaseAgent
from beeai_framework.agents.types import AgentMeta
from beeai_framework.backend.message import AnyMessage, AssistantMessage, Role
from beeai_framework.context import Run, RunContext
from beeai_framework.emitter.emitter import Emitter
from beeai_framework.errors import FrameworkError
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.memory.base_memory import BaseMemory


class EchoAgentRunOutput(BaseModel):
    message: InstanceOf[AnyMessage]


# This is a simple echo agent that echoes back the last message it received.
class EchoAgent(BaseAgent[EchoAgentRunOutput]):
    memory: BaseMemory | None = None

    def __init__(self, memory: BaseMemory) -> None:
        super().__init__()
        self.memory = memory

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["agent", "custom"],
            creator=self,
        )

    def run(
        self,
        input: list[AnyMessage] | None = None,
    ) -> Run[EchoAgentRunOutput]:
        async def handler(context: RunContext) -> EchoAgentRunOutput:
            assert self.memory is not None
            if input:
                await self.memory.add_many(input)
            return EchoAgentRunOutput(message=AssistantMessage(self.memory.messages[-1].text))

        return self._to_run(handler, signal=None)

    @property
    def meta(self) -> AgentMeta:
        return AgentMeta(
            name="EchoAgent",
            description="Simple echo agent.",
            tools=[],
        )


def main() -> None:
    # Create a custom agent factory for the EchoAgent
    def agent_factory(agent: EchoAgent) -> AcpAgent:
        """Factory method to create an AcpAgent from a EchoAgent."""

        async def run(
            input: list[acp_models.Message], context: acp_context.Context
        ) -> AsyncGenerator[acp_types.RunYield, acp_types.RunYieldResume]:
            framework_messages = [
                to_framework_message(Role(message.parts[0].role), str(message))  # type: ignore[attr-defined]
                for message in input
            ]
            response = await agent.run(framework_messages)
            yield acp_models.MessagePart(content=response.message.text, role="assistant")  # type: ignore[call-arg]

        # Create an AcpAgent instance with the run function
        return AcpAgent(fn=run, name=agent.meta.name, description=agent.meta.description)

    # Register the custom agent factory with the ACP server
    AcpAgentServer.register_factory(EchoAgent, agent_factory)
    # Create an instance of the EchoAgent with UnconstrainedMemory
    agent = EchoAgent(memory=UnconstrainedMemory())
    # Register the agent with the ACP server and run the HTTP server
    AcpAgentServer().register([agent]).serve()


if __name__ == "__main__":
    try:
        main()
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

# run: beeai agent run EchoAgent "Hello"
