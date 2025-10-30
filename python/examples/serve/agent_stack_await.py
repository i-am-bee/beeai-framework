from beeai_framework.adapters.agentstack.serve.server import AgentStackServer
from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.agents.requirement.requirements.ask_permission import AskPermissionRequirement
from beeai_framework.backend import ChatModel
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware
from beeai_framework.tools.weather import OpenMeteoTool

try:
    from agentstack_sdk.a2a.extensions.ui.agent_detail import AgentDetail
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [agentstack] not found.\nRun 'pip install \"beeai-framework[agentstack]\"' to install."
    ) from e


def main() -> None:
    llm = ChatModel.from_name("ollama:granite4:micro")
    agent = RequirementAgent(
        llm=llm,
        tools=[OpenMeteoTool()],
        requirements=[AskPermissionRequirement(include=OpenMeteoTool)],
        name="Framework weather await agent",
        description="Weather agent that asks for a permission before using a tool!",
        middlewares=[GlobalTrajectoryMiddleware()],
    )

    server = AgentStackServer()
    server.register(agent, detail=AgentDetail(interaction_mode="multi-turn"))
    server.serve()


if __name__ == "__main__":
    main()
