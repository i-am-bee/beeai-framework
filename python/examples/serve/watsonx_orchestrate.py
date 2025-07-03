from beeai_framework.adapters.watsonx_orchestrate import WatsonxOrchestrateServer, WatsonxOrchestrateServerConfig
from beeai_framework.agents.experimental import RequirementAgent
from beeai_framework.backend import ChatModel
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.weather import OpenMeteoTool


def main() -> None:
    llm = ChatModel.from_name("ollama:granite3.3:8b")
    agent = RequirementAgent(llm=llm, tools=[OpenMeteoTool()], memory=UnconstrainedMemory(), role="a weather agent")

    config = WatsonxOrchestrateServerConfig(port=8080, host="0.0.0.0", api_key=None)  # optional
    server = WatsonxOrchestrateServer(config=config)
    server.register(agent)

    # start an API with /chat/completions endpoint which is compatible with Watsonx Orchestrate
    server.serve()


if __name__ == "__main__":
    main()
