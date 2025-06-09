from beeai_framework.adapters.beeai_platform.serve.server import BeeAIPlatformServer
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

    # Runs HTTP server that registers to BeeAI platform
    server = BeeAIPlatformServer()
    server.register(
        agent,
        name="granite_chat_agent",
        description="Simple chat agent",  # (optional)
        ui={"type": "chat"},  # default is chat (optional)
        tags=["example"],  # (optional)
        recommended_models=["granite3.3:8b"],  # (optional)
    )
    server.serve()


if __name__ == "__main__":
    main()

# run: beeai agent run chat_agent
