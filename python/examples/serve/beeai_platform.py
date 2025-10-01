from beeai_framework.adapters.beeai_platform.backend.chat import BeeAIPlatformChatModel
from beeai_framework.adapters.beeai_platform.context import BeeAIPlatformContext
from beeai_framework.adapters.beeai_platform.serve.server import BeeAIPlatformMemoryManager, BeeAIPlatformServer
from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.context import RunContext, RunMiddlewareProtocol

try:
    from beeai_sdk.a2a.extensions.ui.agent_detail import AgentDetail
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [beeai-platform] not found.\nRun 'pip install \"beeai-framework[beeai-platform]\"' to install."
    ) from e


def main() -> None:
    class PlatformMiddleware(RunMiddlewareProtocol):
        def bind(self, ctx: RunContext) -> None:
            platform_ctx = BeeAIPlatformContext.get()
            print(platform_ctx.extensions["form"])

    agent = RequirementAgent(
        llm=BeeAIPlatformChatModel(preferred_models=["ollama/granite3.3:8b", "openai/gpt-5"]),
        middlewares=[PlatformMiddleware()],
        # tools=[DuckDuckGoSearchTool(), OpenMeteoTool()],
        # memory=UnconstrainedMemory(),
        # middlewares=[GlobalTrajectoryMiddleware()],
    )

    # Runs HTTP server that registers to BeeAI platform
    server = BeeAIPlatformServer(config={"configure_telemetry": False}, memory_manager=BeeAIPlatformMemoryManager())
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
