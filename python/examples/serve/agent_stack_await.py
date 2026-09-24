from beeai_framework.adapters.agentstack.backend.chat import AgentStackChatModel
from beeai_framework.adapters.agentstack.serve.server import AgentStackServer
from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.agents.requirement.requirements.ask_permission import AskPermissionRequirement
from beeai_framework.backend import ChatModelParameters
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware
from beeai_framework.tools.weather import OpenMeteoTool


def main() -> None:
    agent = RequirementAgent(
        llm=AgentStackChatModel(parameters=ChatModelParameters(stream=True)),
        tools=[OpenMeteoTool()],
        requirements=[AskPermissionRequirement(include=OpenMeteoTool)],
        name="Framework weather await agent",
        description="Weather agent that asks for a permission before using a tool!",
        middlewares=[GlobalTrajectoryMiddleware()],
    )

    server = AgentStackServer(config={"configure_telemetry": False})
    server.register(agent)
    server.serve()


if __name__ == "__main__":
    main()
