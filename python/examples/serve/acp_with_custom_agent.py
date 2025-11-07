import sys
import traceback
from collections.abc import AsyncGenerator
from typing import Unpack

import acp_sdk.models as acp_models
import acp_sdk.server.context as acp_context
import acp_sdk.server.types as acp_types

from beeai_framework.adapters.acp import ACPServer
from beeai_framework.adapters.acp.serve._utils import acp_msgs_to_framework_msgs
from beeai_framework.adapters.acp.serve.agent import ACPServerAgent
from beeai_framework.adapters.acp.serve.server import ACPServerMetadata, to_acp_agent_metadata
from beeai_framework.agents import AgentOptions, AgentOutput, BaseAgent
from beeai_framework.backend.message import AnyMessage, AssistantMessage, UserMessage
from beeai_framework.emitter.emitter import Emitter
from beeai_framework.errors import FrameworkError
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.memory.base_memory import BaseMemory
from beeai_framework.runnable import runnable_entry


# This is a simple echo agent that echoes back the last message it received.
class EchoAgent(BaseAgent):
    memory: BaseMemory

    def __init__(self, memory: BaseMemory) -> None:
        super().__init__()
        self.memory = memory

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["agent", "custom"],
            creator=self,
        )

    @runnable_entry
    async def run(self, input: str | list[AnyMessage], /, **kwargs: Unpack[AgentOptions]) -> AgentOutput:
        assert self.memory is not None

        if isinstance(input, str):
            await self.memory.add(UserMessage(input))
        elif isinstance(input, list):
            await self.memory.add_many(input)

        text_input = self.memory.messages[-1].text if self.memory.messages else ""
        return AgentOutput(output=[AssistantMessage(text_input)])


def main() -> None:
    # Create a custom agent factory for the EchoAgent
    def agent_factory(agent: EchoAgent, *, metadata: ACPServerMetadata | None = None) -> ACPServerAgent:
        """Factory method to create an ACPAgent from a EchoAgent."""
        if metadata is None:
            metadata = {}

        async def run(
            input: list[acp_models.Message], context: acp_context.Context
        ) -> AsyncGenerator[acp_types.RunYield, acp_types.RunYieldResume]:
            framework_messages = acp_msgs_to_framework_msgs(input)
            response = await agent.run(framework_messages)
            yield acp_models.MessagePart(content=response.last_message.text, role="assistant")  # type: ignore[call-arg]

        # Create an ACPAgent instance with the run function
        return ACPServerAgent(
            fn=run,
            name=metadata.get("name", agent.meta.name),
            description=metadata.get("description", agent.meta.description),
            metadata=to_acp_agent_metadata(metadata),
        )

    # Register the custom agent factory with the ACP server
    ACPServer.register_factory(EchoAgent, agent_factory)
    # Create an instance of the EchoAgent with UnconstrainedMemory
    agent = EchoAgent(memory=UnconstrainedMemory())
    # Register the agent with the ACP server and run the HTTP server
    ACPServer().register(agent, name="echo_agent").serve()


if __name__ == "__main__":
    try:
        main()
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

# run: beeai agent run echo_agent "Hello"
