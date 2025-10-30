from beeai_framework.adapters.a2a import A2AServer, A2AServerConfig
from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.backend import ChatModel
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.serve.utils import LRUMemoryManager
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool
from beeai_framework.tools.weather import OpenMeteoTool


def main() -> None:
    llm = ChatModel.from_name("ollama:granite4:micro")
    agent = RequirementAgent(
        llm=llm,
        tools=[DuckDuckGoSearchTool(), OpenMeteoTool()],
        memory=UnconstrainedMemory(),
    )

    # Register the agent with the A2A server and run the HTTP server
    # For the ToolCallingAgent, we don't need to specify A2AAgent factory method
    # because it is already registered in the A2AServer
    # we use LRU memory manager to keep limited amount of sessions in the memory
    A2AServer(
        config=A2AServerConfig(port=9999, protocol="jsonrpc"), memory_manager=LRUMemoryManager(maxsize=100)
    ).register(agent, send_trajectory=True).serve()


if __name__ == "__main__":
    main()
