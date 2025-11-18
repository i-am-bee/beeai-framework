import sys
import traceback
from collections.abc import AsyncGenerator

import acp_sdk.models as acp_models
import acp_sdk.server.context as acp_context
import acp_sdk.server.types as acp_types
from pydantic import BaseModel, InstanceOf

from beeai_framework.adapters.acp import ACPServer
from beeai_framework.adapters.acp.serve._utils import acp_msgs_to_framework_msgs
from beeai_framework.adapters.acp.serve.agent import ACPServerAgent
from beeai_framework.adapters.acp.serve.server import to_acp_agent_metadata
from beeai_framework.adapters.beeai_platform.serve.server import BeeAIPlatformServerMetadata
from beeai_framework.agents.base import BaseAgent
from beeai_framework.backend.message import AnyMessage, AssistantMessage
from beeai_framework.context import Run, RunContext
from beeai_framework.emitter.emitter import Emitter
from beeai_framework.errors import FrameworkError
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.memory.base_memory import BaseMemory


class EchoAgentRunOutput(BaseModel):
    message: InstanceOf[AnyMessage]


# This is a simple echo agent that echoes back the last message it received.
class EchoAgent(BaseAgent[EchoAgentRunOutput]):
    memory: BaseMemory

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


def main() -> None:
    # Create a custom agent factory for the EchoAgent
    def agent_factory(agent: EchoAgent, *, metadata: BeeAIPlatformServerMetadata | None = None) -> ACPServerAgent:
        """Factory method to create an ACPAgent from a EchoAgent."""
        if metadata is None:
            metadata = {}

        async def run(
            input: list[acp_models.Message], context: acp_context.Context
        ) -> AsyncGenerator[acp_types.RunYield, acp_types.RunYieldResume]:
            framework_messages = acp_msgs_to_framework_msgs(input)
            response = await agent.run(framework_messages)
            yield acp_models.MessagePart(content=response.message.text, role="assistant")  # type: ignore[call-arg]

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
    # Enamble self-registration for the agent to BeeAI platform
    ACPServer(config={"self_registration": True}).register(agent, name="echo_agent").serve()


if __name__ == "__main__":
    try:
        main()
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

# run: beeai agent run echo_agent "Hello"
