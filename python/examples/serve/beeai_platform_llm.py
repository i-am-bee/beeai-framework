from beeai_framework.adapters.beeai_platform.serve.server import BeeAIPlatformServer
from beeai_framework.backend.chat import ChatModel
from beeai_framework.serve.utils import UnlimitedMemoryManager


def main() -> None:
    llm = ChatModel.from_name("ollama:granite4:micro")

    # Runs HTTP server that registers to BeeAI platform
    server = BeeAIPlatformServer(config={"configure_telemetry": False}, memory_manager=UnlimitedMemoryManager())
    server.register(llm, name="Ollama model")
    server.serve()


if __name__ == "__main__":
    main()

# run: beeai agent run chat_agent
