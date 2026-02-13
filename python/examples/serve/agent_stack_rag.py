from typing import Annotated

from emitter import EventMeta
from emitter.utils import create_internal_event_matcher

from beeai_framework.adapters.agentstack.backend.chat import AgentStackChatModel
from beeai_framework.adapters.agentstack.backend.embedding import AgentstackEmbeddingModel
from beeai_framework.adapters.agentstack.backend.vector_store import NativeVectorStore
from beeai_framework.adapters.agentstack.serve.server import AgentStackMemoryManager, AgentStackServer
from beeai_framework.adapters.agentstack.serve.types import BaseAgentStackExtensions
from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.backend import ChatModelParameters
from beeai_framework.backend.types import Document
from beeai_framework.context import RunContext, RunContextStartEvent, RunMiddlewareProtocol
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.search.retrieval import VectorStoreSearchTool

try:
    from agentstack_sdk.a2a.extensions import EmbeddingServiceExtensionServer, EmbeddingServiceExtensionSpec
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [agentstack] not found.\nRun 'pip install \"beeai-framework[agentstack]\"' to install."
    ) from e


# The middleware is necessary since the embedding service's initialization occurs after the client's call to the agent.
class RAGMiddleware(RunMiddlewareProtocol):
    def __init__(self, vector_store: NativeVectorStore) -> None:
        self._vector_store = vector_store

    def bind(self, ctx: RunContext) -> None:  # pyrefly: ignore [bad-override]
        # Only insert the documents during initialization.
        ctx.emitter.on(create_internal_event_matcher("start"), self._on_start)

    async def _on_start(self, _: RunContextStartEvent, meta: EventMeta) -> None:
        if not self._vector_store.is_initialized:
            print("debug: initializing vector store")
            await self._vector_store.add_documents(
                [
                    Document(content="My name is John.", metadata={}),
                    Document(content="I am a python programmer.", metadata={}),
                    Document(content="I am 30 years old.", metadata={}),
                ]
            )


def main() -> None:
    llm = AgentStackChatModel(
        preferred_models=["openai:gpt-4o", "ollama:llama3.1:8b"],
        parameters=ChatModelParameters(stream=True),
    )

    # Initialize the embedding model from the Agent Stack.
    embedding_model = AgentstackEmbeddingModel(preferred_models=["ollama:nomic-embed-text:latest"])

    vector_store = NativeVectorStore(embedding_model)
    # vector_store = VectorStore.from_name("AgentStack:NativeVectorStore", embedding_model=embedding_model)

    agent = RequirementAgent(
        llm=llm,
        tools=[VectorStoreSearchTool(vector_store)],
        memory=UnconstrainedMemory(),
        middlewares=[RAGMiddleware(vector_store)],  # add middleware to initialize vector store
    )

    # define custom extensions
    class CustomExtensions(BaseAgentStackExtensions):
        # "The property name must be 'embedding'.
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
