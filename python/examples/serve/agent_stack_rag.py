from typing import Annotated

from beeai_framework.adapters.agentstack.backend.chat import AgentStackChatModel
from beeai_framework.adapters.agentstack.backend.embedding import AgentstackEmbeddingModel
from beeai_framework.adapters.agentstack.backend.vector_store import NativeVectorStore
from beeai_framework.adapters.agentstack.serve.server import AgentStackMemoryManager, AgentStackServer
from beeai_framework.adapters.agentstack.serve.types import BaseAgentStackExtensions
from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.backend import ChatModelParameters
from beeai_framework.backend.types import Document
from beeai_framework.context import RunContext, RunMiddlewareProtocol
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.search.retrieval import VectorStoreSearchTool

try:
    from agentstack_sdk.a2a.extensions import EmbeddingServiceExtensionServer, EmbeddingServiceExtensionSpec
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [agentstack] not found.\nRun 'pip install \"beeai-framework[agentstack]\"' to install."
    ) from e


def main() -> None:
    llm = AgentStackChatModel(
        preferred_models=["openai:gpt-4o", "ollama:llama3.1:8b"],
        parameters=ChatModelParameters(stream=True),
    )

    # setup embedding model from the agent Stack
    embedding_model = AgentstackEmbeddingModel(preferred_models=["ollama:nomic-embed-text:latest"])

    vector_store = NativeVectorStore(embedding_model)

    class RAGMiddleware(RunMiddlewareProtocol):
        async def bind(self, ctx: RunContext) -> None:  # pyrefly: ignore [bad-override]
            # insert the documents only at the beginning
            if vector_store._vector_store is None:
                print("debug: initializing vector store")
                await vector_store.add_documents([Document(content="My name is John.", metadata={})])
                await vector_store.add_documents([Document(content="I am a python programmer.", metadata={})])
                await vector_store.add_documents([Document(content="I am 30 years old.", metadata={})])

    agent = RequirementAgent(
        llm=llm,
        tools=[VectorStoreSearchTool(vector_store)],
        memory=UnconstrainedMemory(),
        middlewares=[RAGMiddleware()],  # add middleware to initialize vector store
    )

    # define custom extensions
    class CustomExtensions(BaseAgentStackExtensions):
        # name of the property must be 'embedding'
        embedding: Annotated[
            EmbeddingServiceExtensionServer,
            EmbeddingServiceExtensionSpec.single_demand(suggested=tuple(embedding_model.preferred_models)),
        ]

    # Runs HTTP server that registers to Agent Stack
    server = AgentStackServer(memory_manager=AgentStackMemoryManager())
    server.register(agent, name="Framework RAG agent", extensions=CustomExtensions)
    server.serve()


if __name__ == "__main__":
    main()
