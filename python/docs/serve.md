# 🚚️ Serve

## Overview

The `Serve` module enables developers to expose components built with the BeeAI Framework through a server to external clients.
Out of the box, we provide implementations for protocols such as ACP and MCP, allowing you to quickly serve existing functionalities.
You can also create your own custom adapter if needed.

> [!NOTE]
>
> Location within the framework: [beeai_framework/serve](/python/beeai_framework/serve).

--- 

## Support Providers

The following table lists the currently supported providers:

| Name                                                  | Dependency                                   | Location                    |
|-------------------------------------------------------|----------------------------------------------|-----------------------------|
| [`ACP`](https://agentcommunicationprotocol.dev)       | `beeai_framework.adapters.acp.serve`         | `beeai-platform[acp]`       |
| [`BeeAI Platform`](https://beeai.dev/)                | `beeai_framework.adapters.beeai_platform.serve` | `beeai-platform[beeai-platform]` |
| [`MCP`](https://modelcontextprotocol.io/) | `beeai_framework.adapters.mcp.serve`         | `beeai-platform[mcp]`       |


For more details, see the [integration page](integrations.md).

## Usage

```python
from beeai_framework.adapters.acp import ACPServer, ACPServerConfig
from beeai_framework.agents.tool_calling.agent import ToolCallingAgent
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.backend import ChatModel

 # creates an agent
agent = ToolCallingAgent(llm=ChatModel.from_name("ollama:granite3.3"), tools=[], memory=UnconstrainedMemory())

server = ACPServer(config=ACPServerConfig(port=8001)) # creates a server
server.register(agent) # register the agent
server.serve() # spawns a server
```

## Extending functionality

By default, each provider supports registration of a limited set of modules (agents, tools, templates, etc.).
You can extend this functionality by registering a custom factory using `Server.register_factory` method.