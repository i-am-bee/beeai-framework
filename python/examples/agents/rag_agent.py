import asyncio
import logging
import os
import sys
import traceback

from beeai_framework.adapters.beeai.backend.document_processor import LLMDocumentReranker
from beeai_framework.adapters.beeai.backend.vector_store import TemporalVectorStore
from beeai_framework.adapters.langchain.backend.vector_store import LangChainVectorStore
from beeai_framework.adapters.langchain.mappers.documents import lc_document_to_document
from beeai_framework.agents.experimental.rag import RAGAgent, RagAgentRunInput
from beeai_framework.backend import UserMessage
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.embedding import EmbeddingModel
from beeai_framework.backend.vector_store import VectorStore
from beeai_framework.errors import FrameworkError
from beeai_framework.logger import Logger
from beeai_framework.memory import UnconstrainedMemory

# LC dependencies - to be swapped with BAI dependencies
try:
    from langchain_community.document_loaders import UnstructuredMarkdownLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional modules are not found.\nRun 'pip install \"beeai-framework[rag]\"' to install."
    ) from e


from dotenv import load_dotenv

load_dotenv()  # load environment variables
logger = Logger("rag-agent", level=logging.DEBUG)


POPULATE_VECTOR_DB = True
VECTOR_DB_PATH_4_DUMP = ""  # Set this path for persistency
INPUT_DOCUMENTS_LOCATION = "docs/integrations"


async def populate_documents() -> VectorStore | None:
    embedding_model = EmbeddingModel.from_name("watsonx:ibm/slate-125m-english-rtrvr-v2", truncate_input_tokens=500)

    # Load existing vector store if available
    if VECTOR_DB_PATH_4_DUMP and os.path.exists(VECTOR_DB_PATH_4_DUMP):
        print(f"Loading vector store from: {VECTOR_DB_PATH_4_DUMP}")
        preloaded_vector_store: VectorStore = TemporalVectorStore.load(
            path=VECTOR_DB_PATH_4_DUMP, embedding_model=embedding_model
        )
        return preloaded_vector_store

    # Create new vector store if population is enabled
    if POPULATE_VECTOR_DB:
        loader = UnstructuredMarkdownLoader(file_path="docs/modules/agents.mdx")
        try:
            docs = loader.load()
        except Exception:
            return None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=1000)
        all_splits = text_splitter.split_documents(docs)
        documents = [lc_document_to_document(document) for document in all_splits]
        print(f"Loaded {len(documents)} documents")

        print("Rebuilding vector store")
        # Adapter example
        vector_store: TemporalVectorStore = VectorStore.from_name(
            name="beeai:TemporalVectorStore", embedding_model=embedding_model
        )  # type: ignore[assignment]
        # Native examples
        # vector_store: TemporalVectorStore = TemporalVectorStore(embedding_model=embedding_model)
        # vector_store = InMemoryVectorStore(embedding_model)
        _ = await vector_store.add_documents(documents=documents)
        if VECTOR_DB_PATH_4_DUMP and isinstance(vector_store, LangChainVectorStore):
            print(f"Dumping vector store to: {VECTOR_DB_PATH_4_DUMP}")
            vector_store.vector_store.dump(VECTOR_DB_PATH_4_DUMP)
        return vector_store

    # Neither existing DB found nor population enabled
    return None


async def main() -> None:
    vector_store = await populate_documents()
    if vector_store is None:
        raise FileNotFoundError(
            f"Vector database not found at {VECTOR_DB_PATH_4_DUMP}. "
            "Either set POPULATE_VECTOR_DB=True to create a new one, or ensure the database file exists."
        )

    llm = ChatModel.from_name("ollama:llama3.2")
    reranker = LLMDocumentReranker(llm)

    agent = RAGAgent(llm=llm, memory=UnconstrainedMemory(), vector_store=vector_store, reranker=reranker)

    response = await agent.run(RagAgentRunInput(message=UserMessage("What agents are available in BeeAI?")))
    print(response.message.text)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
