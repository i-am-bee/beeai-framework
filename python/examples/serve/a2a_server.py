from beeai_framework.adapters.a2a import A2AServer, A2AServerConfig
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
    )

    # Register the agent with the A2A server and run the HTTP server
    # For the ToolCallingAgent, we dont need to specify ACPAgent factory method
    # because it is already registered in the A2AServer
    A2AServer(config=A2AServerConfig(port=9999)).register(agent).serve()


if __name__ == "__main__":
    main()
