# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from beeai_framework.adapters.beeai_platform.serve.server import BeeAIPlatformServer
from beeai_framework.agents.experimental import RequirementAgent
from beeai_framework.agents.experimental.requirements.ask_permission import AskPermissionRequirement
from beeai_framework.backend import ChatModel
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware
from beeai_framework.tools.weather import OpenMeteoTool


def main() -> None:
    llm = ChatModel.from_name("ollama:granite3.3:8b")
    agent = RequirementAgent(
        llm=llm,
        tools=[OpenMeteoTool()],
        requirements=[AskPermissionRequirement(include=OpenMeteoTool)],
        name="my_weather_await_agent",
        description="Weather agent that asks for a permission before using a tool!",
        middlewares=[GlobalTrajectoryMiddleware()],
    )

    server = BeeAIPlatformServer()
    server.register(agent, ui={"type": "chat"})
    server.serve()


if __name__ == "__main__":
    main()
