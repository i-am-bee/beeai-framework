from beeai_framework.adapters.agentstack.serve.server import AgentStackServer
from beeai_framework.backend.chat import ChatModel
from beeai_framework.serve.utils import UnlimitedMemoryManager


def main() -> None:
    llm = ChatModel.from_name("ollama:granite4:micro")

    # Runs HTTP server that registers to Agent Stack
    server = AgentStackServer(config={"configure_telemetry": False}, memory_manager=UnlimitedMemoryManager())
    server.register(llm, name="Ollama model")
    server.serve()


if __name__ == "__main__":
    main()

# run: beeai agent run chat_agent
