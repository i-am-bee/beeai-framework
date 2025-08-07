import asyncio
import os

from beeai_framework.agents.experimental import RequirementAgent
from beeai_framework.backend import ChatModel
from beeai_framework.backend.document_loader import DocumentLoader
from beeai_framework.backend.embedding import EmbeddingModel
from beeai_framework.backend.text_splitter import TextSplitter
from beeai_framework.backend.vector_store import VectorStore
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware
from beeai_framework.tools import Tool
from beeai_framework.tools.search.retrieval import VectorStoreSearchTool

POPULATE_VECTOR_DB = True
VECTOR_DB_PATH_4_DUMP = ""  # Set this path for persistency


async def setup_vector_store() -> VectorStore | None:
    """
    Setup vector store with BeeAI framework documentation.
    """
    embedding_model = EmbeddingModel.from_name("watsonx:ibm/slate-125m-english-rtrvr-v2", truncate_input_tokens=500)

    # Load existing vector store if available
    if VECTOR_DB_PATH_4_DUMP and os.path.exists(VECTOR_DB_PATH_4_DUMP):
        print(f"Loading vector store from: {VECTOR_DB_PATH_4_DUMP}")
        from beeai_framework.adapters.beeai.backend.vector_store import TemporalVectorStore

        preloaded_vector_store: VectorStore = TemporalVectorStore.load(
            path=VECTOR_DB_PATH_4_DUMP, embedding_model=embedding_model
        )
        return preloaded_vector_store

    # Create new vector store if population is enabled
    # NOTE: Vector store population is typically done offline in production applications
    if POPULATE_VECTOR_DB:
        # Load documentation about BeeAI agents - this serves as our knowledge base
        # for answering questions about the different types of agents available
        loader = DocumentLoader.from_name(
            name="langchain:UnstructuredMarkdownLoader", file_path="docs/modules/agents.mdx"
        )
        try:
            documents = await loader.load()
        except Exception as e:
            print(f"Failed to load documents: {e}")
            return None

        # Split documents into chunks
        text_splitter = TextSplitter.from_name(
            name="langchain:RecursiveCharacterTextSplitter", chunk_size=1000, chunk_overlap=200
        )
        documents = await text_splitter.split_documents(documents)
        print(f"Loaded {len(documents)} document chunks")

        # Create vector store and add documents
        vector_store = VectorStore.from_name(name="beeai:TemporalVectorStore", embedding_model=embedding_model)
        await vector_store.add_documents(documents=documents)
        print("Vector store populated with documents")

        return vector_store

    return None


async def main() -> None:
    """
    Example demonstrating RequirementAgent using VectorStoreSearchTool.

    The agent will use the vector store search tool to find relevant information
    about BeeAI framework agents and provide comprehensive answers.

    Note: In typical applications, you would use a pre-populated vector store
    rather than populating it at runtime. This example includes population
    logic for demonstration purposes only.
    """
    # Setup vector store with BeeAI documentation
    vector_store = await setup_vector_store()
    if vector_store is None:
        raise FileNotFoundError(
            "Failed to instantiate Vector Store. "
            "Either set POPULATE_VECTOR_DB=True to create a new one, or ensure the database file exists."
        )

    # Create the vector store search tool
    search_tool = VectorStoreSearchTool(vector_store=vector_store)

    # Alternative: Create search tool using dynamic loading
    # embedding_model = EmbeddingModel.from_name("watsonx:ibm/slate-125m-english-rtrvr-v2", truncate_input_tokens=500)
    # search_tool = VectorStoreSearchTool.from_name(
    #     name="beeai:TemporalVectorStore",
    #     embedding_model=embedding_model
    # )

    # Create RequirementAgent with the vector store search tool
    llm = ChatModel.from_name("ollama:llama3.1:8b")
    agent = RequirementAgent(
        llm=llm,
        memory=UnconstrainedMemory(),
        instructions=(
            "You are a helpful assistant that answers questions about the BeeAI framework. "
            "Use the vector store search tool to find relevant information from the documentation "
            "before providing your answer. Always search for information first, then provide a "
            "comprehensive response based on what you found."
        ),
        tools=[search_tool],
        # Log all tool calls to the console for easier debugging
        middlewares=[GlobalTrajectoryMiddleware(included=[Tool])],
    )

    query = "What types of agents are available in BeeAI?"
    response = await agent.run(query)
    print(f"query: {query}\nResponse: {response.answer.text}")


if __name__ == "__main__":
    asyncio.run(main())
