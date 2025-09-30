from beeai_framework.adapters.mcp.serve.server import MCPServer, MCPServerConfig
from beeai_framework.agents.experimental import RequirementAgent
from beeai_framework.backend import ChatModel
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool
from beeai_framework.tools.weather import OpenMeteoTool


def main() -> None:
    llm = ChatModel.from_name("ollama:granite3.3:8b")
    agent = RequirementAgent(
        llm=llm,
        tools=[DuckDuckGoSearchTool(), OpenMeteoTool()],
        memory=UnconstrainedMemory(),
        middlewares=[GlobalTrajectoryMiddleware()],
    )

    # All Runnables are supported, including Agents, ChatModels, Tools, and custom Runnables
    MCPServer(config=MCPServerConfig(transport="streamable-http")).register_many([agent, llm]).serve()


if __name__ == "__main__":
    main()
