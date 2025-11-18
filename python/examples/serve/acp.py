from beeai_framework.adapters.acp import ACPServer, ACPServerConfig
from beeai_framework.agents.experimental import RequirementAgent
from beeai_framework.backend import ChatModel
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool
from beeai_framework.tools.weather import OpenMeteoTool


def main() -> None:
    llm = ChatModel.from_name("ollama:granite3.3:8b")
    agent = RequirementAgent(
        llm=llm,
        tools=[DuckDuckGoSearchTool(), OpenMeteoTool()],
        memory=UnconstrainedMemory(),
        # specify the agent's name and other metadata
        name="my_agent",
        description="A simple agent",
    )

    # Register the agent with the ACP server and run the HTTP server
    # For the ToolCallingAgent and ReActAgent, we dont need to specify ACPAgent factory method
    # because they are already registered in the ACPServer
    ACPServer(config=ACPServerConfig(port=8001)).register(agent, tags=["example"]).serve()


if __name__ == "__main__":
    main()
