# 🔌 Integrations

<!-- TOC -->
## Table of Contents
- [Agent Communication Protocol (ACP) Integration](#agent-communication-protocol-integration)
  - [ACPAgent](#acp-agent)
  - [ACPServer](#acp-server)
- [BeeAI Platform Integration](#beeai-platform-integration)
  - [BeeAIPlatformAgent](#beeai-platform-agent)
  - [BeeAIPlatformServer](#beeai-platform-server)
- [Model Context Protocol (MCP) Integration](#model-context-protocol-integration)
  - [MCPTool](#mcp-tool)
  - [MCPServer](#mcp-server)
- [Agent2Agent (A2A) Protocol Integration](#agent2agent-protocol-integration)
  - [A2AAgent](#a2a-agent)
  - [A2AServer](#a2a-server)
- [Examples](#examples)
<!-- /TOC -->

---

## Agent Communication Protocol Integration

### ACP Agent

ACPAgent lets you easily connect with external agents using the [Agent Communication Protocol (ACP)](https://agentcommunicationprotocol.dev/). ACP is a standard for agent-to-agent communication, allowing different AI agents to interact regardless of how they’re built. This agent works with any ACP-compliant service.

Use ACPAgent When:
- You're connecting to your own custom ACP server
- You're developing a multi-agent system where agents communicate via ACP
- You're integrating with a third-party ACP-compliant service that isn't the BeeAI Platform

<!-- embedme examples/agents/providers/acp.py -->

```py
import asyncio
import sys
import traceback

from beeai_framework.adapters.acp.agents import ACPAgent
from beeai_framework.errors import FrameworkError
from beeai_framework.memory.unconstrained_memory import UnconstrainedMemory
from examples.helpers.io import ConsoleReader


async def main() -> None:
    reader = ConsoleReader()

    agent = ACPAgent(agent_name="chat", url="http://127.0.0.1:8000", memory=UnconstrainedMemory())
    for prompt in reader:
        # Run the agent and observe events
        response = await agent.run(prompt).on(
            "update",
            lambda data, event: (reader.write("Agent 🤖 (debug) : ", data)),
        )

        reader.write("Agent 🤖 : ", response.result.text)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
```

_Source: [examples/agents/providers/acp.py](/python/examples/agents/providers/acp.py)_

The availability of ACP agents depends on the server you're connecting to. You can check which agents are available by using the check_agent_exists method:

```py
try:
    await agent.check_agent_exists()
    print("Agent exists and is available")
except AgentError as e:
    print(f"Agent not available: {e.message}")
```

If you need to create your own ACP server with custom agents, BeeAI framework provides the AcpServer class.

### ACP Server

Basic example:

<!-- embedme examples/serve/acp.py -->

```py
from beeai_framework.adapters.acp import ACPServer, ACPServerConfig
from beeai_framework.agents.tool_calling.agent import ToolCallingAgent
from beeai_framework.agents.types import AgentMeta
from beeai_framework.backend import ChatModel
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool
from beeai_framework.tools.weather import OpenMeteoTool


def main() -> None:
    llm = ChatModel.from_name("ollama:granite3.3:8b")
    agent = ToolCallingAgent(
        llm=llm,
        tools=[DuckDuckGoSearchTool(), OpenMeteoTool()],
        memory=UnconstrainedMemory(),
        # specify the agent's name and other metadata
        meta=AgentMeta(name="my_agent", description="A simple agent", tools=[]),
    )

    # Register the agent with the ACP server and run the HTTP server
    # For the ToolCallingAgent and ReActAgent, we dont need to specify ACPAgent factory method
    # because they are already registered in the ACPServer
    ACPServer(config=ACPServerConfig(port=8001)).register(agent, tags=["example"]).serve()


if __name__ == "__main__":
    main()

```

_Source: [examples/serve/acp.py](/python/examples/serve/acp.py)_

**Custom agent example:**

<!-- embedme examples/serve/acp_with_custom_agent.py -->

```py
import sys
import traceback
from collections.abc import AsyncGenerator

import acp_sdk.models as acp_models
import acp_sdk.server.context as acp_context
import acp_sdk.server.types as acp_types
from pydantic import BaseModel, InstanceOf

from beeai_framework.adapters.acp import ACPServer, acp_msgs_to_framework_msgs
from beeai_framework.adapters.acp.serve.agent import ACPServerAgent
from beeai_framework.adapters.acp.serve.server import to_acp_agent_metadata
from beeai_framework.adapters.beeai_platform.serve.server import BeeAIPlatformServerMetadata
from beeai_framework.agents.base import BaseAgent
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


def main() -> None:
    # Create a custom agent factory for the EchoAgent
    def agent_factory(agent: EchoAgent, *, metadata: BeeAIPlatformServerMetadata | None = None) -> ACPServerAgent:
        """Factory method to create an ACPAgent from a EchoAgent."""
        if metadata is None:
            metadata = {}

        async def run(
            input: list[acp_models.Message], context: acp_context.Context
        ) -> AsyncGenerator[acp_types.RunYield, acp_types.RunYieldResume]:
            response = await agent.run(acp_msgs_to_framework_msgs(input))
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

```

_Source: [examples/serve/acp_with_custom_agent.py](/python/examples/serve/acp_with_custom_agent.py)_

## BeeAI Platform Integration

### BeeAI Platform Agent

BeeaiPlatformAgent provides specialized integration with the [BeeAI Platform](https://beeai.dev/).

Use BeeAIPlatformAgent When:
- You're connecting specifically to the BeeAI Platform services.
- You want forward compatibility for the BeeAI Platform, no matter which protocol it is based on.


<!-- embedme examples/agents/providers/beeai_platform.py -->

```py
import asyncio
import sys
import traceback

from beeai_framework.adapters.beeai_platform.agents import BeeAIPlatformAgent
from beeai_framework.errors import FrameworkError
from beeai_framework.memory.unconstrained_memory import UnconstrainedMemory
from examples.helpers.io import ConsoleReader


async def main() -> None:
    reader = ConsoleReader()

    agent = BeeAIPlatformAgent(agent_name="chat", url="http://127.0.0.1:8333/api/v1/acp/", memory=UnconstrainedMemory())
    for prompt in reader:
        # Run the agent and observe events
        response = await agent.run(prompt).on(
            "update",
            lambda data, event: (reader.write("Agent 🤖 (debug) : ", data)),
        )

        reader.write("Agent 🤖 : ", response.result.text)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

```

_Source: [examples/agents/providers/beeai_platform.py](/python/examples/agents/providers/beeai_platform.py)_

### BeeAI Platform Server

BeeAIPlatformServer is optimized for seamless integration with the [BeeAI Platform](https://beeai.dev/).


<!-- embedme examples/serve/beeai_platform.py -->

```py
from beeai_framework.adapters.beeai_platform.serve.server import BeeAIPlatformServer
from beeai_framework.agents.tool_calling.agent import ToolCallingAgent
from beeai_framework.backend import ChatModel
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool
from beeai_framework.tools.weather import OpenMeteoTool


def main() -> None:
    llm = ChatModel.from_name("ollama:granite3.3:8b")
    agent = ToolCallingAgent(llm=llm, tools=[DuckDuckGoSearchTool(), OpenMeteoTool()], memory=UnconstrainedMemory())

    # Register the agent with the Beeai platform and run the HTTP server
    # For the ToolCallingAgent and ReActAgent, we dont need to specify BeeAIPlatformAgent factory method
    # because they are already registered in the BeeAIPlatformServer
    BeeAIPlatformServer().register(
        agent, name="chat_agent", description="Simple chat agent", ui={"type": "chat"}
    ).serve()


if __name__ == "__main__":
    main()

# run: beeai agent run chat_agent

```

_Source: [examples/serve/beeai_platform.py](/python/examples/serve/beeai_platform.py)_

## Model Context Protocol Integration

### MCP Tool

MCPTool allows you to consume external tools exposed via MCP protocol. See the [MCP tool documentation](/python/docs/tools.md#mcp-tool) for more information.
 
### MCP Server

MCPServer allows you to expose your tools to external systems that support the Model Context Protocol (MCP) standard, enabling seamless integration with LLM tools ecosystems.

Key benefits
- Fast setup with minimal configuration
- Support for multiple transport options
- Register multiple tools on a single server
- Custom server settings and instructions


<!-- embedme examples/serve/mcp_tool.py -->

```py
from mcp.server.fastmcp.server import Settings

from beeai_framework.adapters.mcp.serve.server import MCPServer, MCPServerConfig
from beeai_framework.tools import tool
from beeai_framework.tools.types import StringToolOutput
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool


@tool
def reverse_tool(word: str) -> StringToolOutput:
    """
    A tool that reverses a word
    """
    return StringToolOutput(result=word[::-1])


def main() -> None:
    # create a MCP server with custom config, register ReverseTool and OpenMeteoTool to the MCP server and run it
    MCPServer(config=MCPServerConfig(transport="sse", settings=Settings(port=8001))).register_many(
        [reverse_tool, OpenMeteoTool()]
    ).serve()


if __name__ == "__main__":
    main()

```

_Source: [examples/serve/mcp_tool.py](/python/examples/serve/mcp_tool.py)_

The MCP adapter uses the MCPServerConfig class to configure the MCP server:

```py
class MCPServerConfig(BaseModel):
    """Configuration for the MCPServer."""
    transport: Literal["stdio", "sse"] = "stdio"  # Transport protocol (stdio or server-sent events)
    name: str = "MCP Server"                     # Name of the MCP server
    instructions: str | None = None              # Optional instructions for the server
    settings: mcp_server.Settings[Any] = Field(default_factory=lambda: mcp_server.Settings())
```

Transport Options
- stdio: Uses standard input/output for communication (default)
- sse: Uses server-sent events over HTTP

Creating an MCP server is easy. You instantiate the MCPServer class with your configuration, register your tools, and then call serve() to start the server:

```py
from beeai_framework.adapters.mcp import MCPServer, MCPServerConfig
from beeai_framework.tools.weather import OpenMeteoTool

# Create an MCP server with default configuration
server = MCPServer()

# Register tools
server.register(OpenMeteoTool())

# Start serving
server.serve()
```

You can configure the server behavior by passing a custom configuration:

```py
from beeai_framework.adapters.mcp import MCPServer
from beeai_framework.tools.weather import OpenMeteoTool
from beeai_framework.tools.search.wikipedia import WikipediaTool
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool

def main():
    server = MCPServer()
    server.register_many([
        OpenMeteoTool(),
        WikipediaTool(),
        DuckDuckGoSearchTool()
    ])
    server.serve()

if __name__ == "__main__":
    main()
```

> [!Tip]
> MCPTool lets you add MCP-compatible tools to any agent, see Tools documentation to learn more.

## Agent2Agent Protocol Integration

### A2A Agent

A2AAgent lets you easily connect with external agents using the [Agent2Agent (A2A)](https://google-a2a.github.io/A2A).

<!-- embedme examples/agents/providers/a2a_agent.py -->

```py
import asyncio
import sys
import traceback

from beeai_framework.adapters.a2a.agents import A2AAgent
from beeai_framework.errors import FrameworkError
from beeai_framework.memory.unconstrained_memory import UnconstrainedMemory
from examples.helpers.io import ConsoleReader


async def main() -> None:
    reader = ConsoleReader()

    agent = A2AAgent(url="http://127.0.0.1:9999", memory=UnconstrainedMemory())
    for prompt in reader:
        # Run the agent and observe events
        response = await agent.run(prompt).on(
            "update",
            lambda data, event: (reader.write("Agent 🤖 (debug) : ", data)),
        )

        reader.write("Agent 🤖 : ", response.result.text)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
```

_Source: [examples/agents/providers/a2a_agent.py](/python/examples/agents/providers/a2a_agent.py)_

### A2A Server

Basic example:

<!-- embedme examples/serve/a2a_server.py#L4-L9999 -->

```py
from beeai_framework.adapters.a2a import A2AServer, A2AServerConfig
from beeai_framework.agents.tool_calling.agent import ToolCallingAgent
from beeai_framework.backend import ChatModel
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool
from beeai_framework.tools.weather import OpenMeteoTool


def main() -> None:
    llm = ChatModel.from_name("ollama:granite3.3:8b")
    agent = ToolCallingAgent(
        llm=llm,
        tools=[DuckDuckGoSearchTool(), OpenMeteoTool()],
        memory=UnconstrainedMemory(),
    )

    # Register the agent with the A2A server and run the HTTP server
    # For the ToolCallingAgent, we dont need to specify ACPAgent factory method
    # because it is already registered in the A2AServer
    A2AServer(config=A2AServerConfig(port=9999)).register(agent).serve()


if __name__ == "__main__":
    main()
```

_Source: [examples/serve/a2a_server.py](/python/examples/serve/a2a_server.py)_

---

## Examples

- All agent examples can be found in [here](/python/examples/agents).
