
# Serve

<!-- TOC -->
## Table of Contents
- [Overview](#overview)
- [Agent Communication Protocol (ACP) Integration](#agent-communication-protocol-integration)
  - [ACPAgent](#acp-agent)
  - [ACPServer](#acp-server)
- [BeeAI Platform Integration](#beeai-platform-integration)
  - [BeeAIPlatformAgent](#beeai-platform-agent)
  - [BeeAIPlatformServer](#beeai-platform-server)
- [Model Context Protocol (MCP) Integration](#model-context-protocol-integration)
  - [MCPServer](#acp-server)
- [Examples](#examples)
<!-- /TOC -->

---

## Overview

AI agents built on large language models (LLMs) provide a structured approach to solving complex problems. Unlike simple LLM interactions, agents can:

- 🔄 Execute multi-step reasoning processes
- 🛠️ Utilize tools to interact with external systems
- 📝 Remember context from previous interactions
- 🔍 Plan and revise their approach based on feedback

Agents control the path to solving a problem, acting on feedback to refine their plan, a capability that improves performance and helps them accomplish sophisticated tasks.

> [!TIP]
>
> For a deeper understanding of AI agents, read this [research article on AI agents and LLMs](https://research.ibm.com/blog/what-are-ai-agents-llm).

> [!NOTE]
>
> Location within the framework: [beeai_framework/agents](/python/beeai_framework/agents).

## Agent Communication Protocol Integration

### ACP Agent

ACPAgent lets you easily connect with external agents using the [Agent Communication Protocol (ACP)](https://agentcommunicationprotocol.dev/). ACP is a standard for agent-to-agent communication, allowing different AI agents to interact regardless of how they’re built. This agent works with any ACP-compliant service.

Use ACPAgent When:
- You're connecting to your own custom ACP server
- You're developing a multi-agent system where agents communicate via ACP
- You're integrating with a third-party ACP-compliant service that isn't the BeeAI Platform

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

```py
from beeai_framework.adapters.acp import AcpAgentServer, AcpServerConfig
from beeai_framework.agents.tool_calling.agent import ToolCallingAgent
from beeai_framework.agents.types import AgentMeta
from beeai_framework.backend import ChatModel
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool
from beeai_framework.tools.weather import OpenMeteoTool


def main() -> None:
    llm = ChatModel.from_name("ollama:granite3.1-dense:8b")
    agent = ToolCallingAgent(
        llm=llm,
        tools=[DuckDuckGoSearchTool(), OpenMeteoTool()],
        memory=UnconstrainedMemory(),
        # specify the agent's name and other metadata
        meta=AgentMeta(name="my_agent", description="A simple agent", tools=[]),
    )

    # Register the agent with the ACP server and run the HTTP server
    # For the ToolCallingAgent and ReActAgent, we dont need to specify AcpAgent factory method
    # because they are already registered in the AcpAgentServer
    AcpAgentServer(config=AcpServerConfig(port=8001)).register(agent).serve()


if __name__ == "__main__":
    main()
```

_Source: [examples/serve/acp.py](/python/examples/serve/acp.py)_

Custom agent example:

```py
import sys
import traceback
from collections.abc import AsyncGenerator

import acp_sdk.models as acp_models
import acp_sdk.server.context as acp_context
import acp_sdk.server.types as acp_types
from pydantic import BaseModel, InstanceOf

from beeai_framework.adapters.acp import AcpAgentServer, acp_msg_to_framework_msg
from beeai_framework.adapters.acp.serve.agent import AcpAgent
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
                acp_msg_to_framework_msg(Role(message.parts[0].role), str(message))  # type: ignore[attr-defined]
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
    AcpAgentServer().register(agent).serve()


if __name__ == "__main__":
    try:
        main()
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

# run: beeai agent run EchoAgent "Hello"
```

_Source: [examples/serve/acp_with_custom_agent.py](/python/examples/serve/acp_with_custom_agent.py)_

## BeeAI Platform Integration

### BeeAI Platform Agent

BeeaiPlatformAgent provides specialized integration with the [BeeAI Platform](https://beeai.dev/).

Use BeeAIPlatformAgent When:
- You're connecting specifically to the BeeAI Platform services.
- You want forward compatibility for the BeeAI Platform, no matter which protocol it is based on.

```py
import asyncio
import sys
import traceback

from beeai_framework.adapters.beeai_platform.agents import BeeaiPlatformAgent
from beeai_framework.errors import FrameworkError
from beeai_framework.memory.unconstrained_memory import UnconstrainedMemory
from examples.helpers.io import ConsoleReader


async def main() -> None:
    reader = ConsoleReader()

    agent = BeeaiPlatformAgent(agent_name="chat", url="http://127.0.0.1:8333/api/v1/acp/", memory=UnconstrainedMemory())
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

```py
from beeai_framework.adapters.beeai_platform.serve.server import BeeaiPlatformServer
from beeai_framework.agents.tool_calling.agent import ToolCallingAgent
from beeai_framework.agents.types import AgentMeta
from beeai_framework.backend import ChatModel
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool
from beeai_framework.tools.weather import OpenMeteoTool


def main() -> None:
    llm = ChatModel.from_name("ollama:granite3.1-dense:8b")
    agent = ToolCallingAgent(
        llm=llm,
        tools=[DuckDuckGoSearchTool(), OpenMeteoTool()],
        memory=UnconstrainedMemory(),
        # specify the agent's name and other metadata
        meta=AgentMeta(name="my_agent", description="A simple agent", tools=[]),
    )

    # Register the agent with the Beeai platform and run the HTTP server
    # For the ToolCallingAgent and ReActAgent, we dont need to specify BeeaiPlatformAgent factory method
    # because they are already registered in the BeeaiPlatformServer
    BeeaiPlatformServer().register(agent).serve()


if __name__ == "__main__":
    main()
```

_Source: [examples/serve/beeai_platform.py](/python/examples/serve/beeai_platform.py)_

## Model Context Protocol Integration

### MCP Server

McpServer allows you to expose your tools to external systems that support the Model Context Protocol (MCP) standard, enabling seamless integration with LLM tools ecosystems.

Key benefits
- Fast setup with minimal configuration
- Support for multiple transport options
- Register multiple tools on a single server
- Custom server settings and instructions

```py
import asyncio
import sys
import traceback

from beeai_framework.adapters.beeai_platform.agents import BeeaiPlatformAgent
from beeai_framework.errors import FrameworkError
from beeai_framework.memory.unconstrained_memory import UnconstrainedMemory
from examples.helpers.io import ConsoleReader


async def main() -> None:
    reader = ConsoleReader()

    agent = BeeaiPlatformAgent(agent_name="chat", url="http://127.0.0.1:8333/api/v1/acp/", memory=UnconstrainedMemory())
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

_Source: [examples/serve/mcp_tool.py](/python/examples/serve/mcp_tool.py)_

The MCP adapter uses the McpServerConfig class to configure the MCP server:

```py
class McpServerConfig(BaseModel):
    """Configuration for the McpServer."""
    transport: Literal["stdio", "sse"] = "stdio"  # Transport protocol (stdio or server-sent events)
    name: str = "MCP Server"                     # Name of the MCP server
    instructions: str | None = None              # Optional instructions for the server
    settings: mcp_server.Settings[Any] = Field(default_factory=lambda: mcp_server.Settings())
```

Transport Options
- stdio: Uses standard input/output for communication (default)
- sse: Uses server-sent events over HTTP

Creating an MCP server is easy. You instantiate the McpServer class with your configuration, register your tools, and then call serve() to start the server:

```py
from beeai_framework.adapters.mcp import McpServer, McpServerConfig
from beeai_framework.tools.weather import OpenMeteoTool

# Create an MCP server with default configuration
server = McpServer()

# Register tools
server.register([OpenMeteoTool()])

# Start serving
server.serve()
```

You can configure the server behavior by passing a custom configuration:

```py
from beeai_framework.adapters.mcp import McpServer
from beeai_framework.tools.weather import OpenMeteoTool
from beeai_framework.tools.search.wikipedia import WikipediaTool
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool

def main():
    server = McpServer()
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

---

## Examples

- All agent examples can be found in [here](/python/examples/agents).
