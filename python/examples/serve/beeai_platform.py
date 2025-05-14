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
