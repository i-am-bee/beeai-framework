from beeai_framework.adapters.openai.serve.server import OpenAIAPIType, OpenAIServer, OpenAIServerConfig
from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.backend import ChatModel
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.weather import OpenMeteoTool


def main() -> None:
    llm = ChatModel.from_name("ollama:granite4:micro")
    agent = RequirementAgent(
        llm=llm,
        tools=[OpenMeteoTool()],
        memory=UnconstrainedMemory(),
    )

    server = OpenAIServer(
        config=OpenAIServerConfig(
            port=9998,
            api=OpenAIAPIType.RESPONSES,
        )
    )
    server.register(agent, name="agent")
    server.register(llm)
    server.serve()


if __name__ == "__main__":
    main()
