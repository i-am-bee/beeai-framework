from beeai_framework.adapters.agentstack.backend.chat import AgentStackChatModel
from beeai_framework.adapters.agentstack.serve.server import AgentStackMemoryManager, AgentStackServer
from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.backend import ChatModelParameters
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool
from beeai_framework.tools.weather import OpenMeteoTool

try:
    from agentstack_sdk.a2a.extensions.ui.agent_detail import AgentDetail
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [agentstack] not found.\nRun 'pip install \"beeai-framework[agentstack]\"' to install."
    ) from e


def main() -> None:
    agent = RequirementAgent(
        llm=AgentStackChatModel(
            preferred_models=["openai:gpt-4o", "ollama:llama3.1:8b"],
            parameters=ChatModelParameters(stream=True),
        ),
        tools=[DuckDuckGoSearchTool(), OpenMeteoTool()],
        memory=UnconstrainedMemory(),
        middlewares=[GlobalTrajectoryMiddleware()],
    )

    # Runs HTTP server that registers to Agent Stack
    server = AgentStackServer(memory_manager=AgentStackMemoryManager())
    server.register(
        agent,
        name="Framework chat agent",  # (optional)
        description="Simple chat agent",  # (optional)
        detail=AgentDetail(interaction_mode="multi-turn"),  # default is multi-turn (optional)
    )
    server.serve()


if __name__ == "__main__":
    main()
