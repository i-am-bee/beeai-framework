from beeai_framework.adapters.beeai_platform.backend.chat import BeeAIPlatformChatModel
from beeai_framework.adapters.beeai_platform.serve.server import BeeAIPlatformMemoryManager, BeeAIPlatformServer
from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.backend import ChatModelParameters
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
    agent = RequirementAgent(
        llm=BeeAIPlatformChatModel(
            preferred_models=["openai:gpt-4o", "ollama:llama3.1:8b"],
            parameters=ChatModelParameters(stream=True),
        ),
        tools=[DuckDuckGoSearchTool(), OpenMeteoTool()],
        memory=UnconstrainedMemory(),
        middlewares=[GlobalTrajectoryMiddleware()],
    )

    # Runs HTTP server that registers to BeeAI platform
    server = BeeAIPlatformServer(memory_manager=BeeAIPlatformMemoryManager())
    server.register(
        agent,
        name="Framework chat agent",  # (optional)
        description="Simple chat agent",  # (optional)
        detail=AgentDetail(interaction_mode="multi-turn"),  # default is multi-turn (optional)
    )
    server.serve()


if __name__ == "__main__":
    main()
