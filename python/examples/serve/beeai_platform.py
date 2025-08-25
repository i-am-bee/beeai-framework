from beeai_framework.adapters.beeai_platform.serve.server import BeeAIPlatformServer
from beeai_framework.agents.experimental import RequirementAgent
from beeai_framework.backend import ChatModel
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool
from beeai_framework.tools.weather import OpenMeteoTool

try:
    from beeai_sdk.a2a.extensions.ui.agent_detail import AgentDetail
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [beeai-platform] not found.\nRun 'pip install \"beeai-framework[beeai-platform]\"' to install."
    ) from e


def main() -> None:
    llm = ChatModel.from_name("ollama:granite3.3:8b")
    agent = RequirementAgent(
        llm=llm,
        tools=[DuckDuckGoSearchTool(), OpenMeteoTool()],
        memory=UnconstrainedMemory(),
        middlewares=[GlobalTrajectoryMiddleware()],
    )

    # Runs HTTP server that registers to BeeAI platform
    server = BeeAIPlatformServer(config={"configure_telemetry": False})
    server.register(
        agent,
        name="Granite chat agent",
        description="Simple chat agent",  # (optional)
        detail=AgentDetail(interaction_mode="multi-turn"),  # default is multi-turn (optional)
    )
    server.serve()


if __name__ == "__main__":
    main()

# run: beeai agent run chat_agent
