from beeai_framework.adapters.acp.server import AcpServer
from beeai_framework.agents.tool_calling.agent import ToolCallingAgent
from beeai_framework.backend import ChatModel
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool
from beeai_framework.tools.weather import OpenMeteoTool


def main() -> None:
    llm = ChatModel.from_name("ollama:granite3.1-dense:8b")
    agent = ToolCallingAgent(llm=llm, tools=[DuckDuckGoSearchTool(), OpenMeteoTool()], memory=UnconstrainedMemory())

    AcpServer().register([agent]).serve()


if __name__ == "__main__":
    main()
