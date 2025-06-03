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
