import asyncio
import os
import sys
import traceback

from beeai_framework.adapters.langchain.mappers.documents import lc_document_to_document
from beeai_framework.adapters.ollama import OllamaChatModel
from beeai_framework.adapters.watsonx.backend.embedding import WatsonxEmbeddingModel
from beeai_framework.agents.rag.agent import RAGAgent, RunInput
from beeai_framework.backend import UserMessage
from beeai_framework.backend.vector_store import VectorStore
from beeai_framework.errors import FrameworkError
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.retrieval.document_processors.document_processors import DocumentsRerankWithLLM

# LC dependencies
try:
    from langchain_community.document_loaders import UnstructuredMarkdownLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional modules are not found.\nRun 'pip install \"beeai-framework[langchain,rag,rag_examples]\"' to install."
    ) from e


from dotenv import load_dotenv

from beeai_framework.retrieval.vector_stores.in_memory_vector_store import InMemoryVectorStore

load_dotenv()  # load environment variables


POPULATE_VECTOR_DB = True
VECTOR_DB_PATH_4_DUMP = ""  # Set this path for persistency
INPUT_DOCUMENTS_LOCATION = "docs-mintlify/integrations"


async def populate_documents() -> VectorStore:
    # Load and chunk contents of the blog
    loader = UnstructuredMarkdownLoader(file_path="python/docs/agents.md")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=1000)
    all_splits = text_splitter.split_documents(docs)
    documents = [lc_document_to_document(document) for document in all_splits]

    embeddings = WatsonxEmbeddingModel(
        model_id="ibm/slate-125m-english-rtrvr-v2",
        project_id=os.getenv("WATSONX_PROJECT_ID"),
        apikey=os.getenv("WATSONX_APIKEY"),
        base_url=os.getenv("WATSONX_URL"),
        truncate_input_tokens=500,
    )

    # Index chunks
    if VECTOR_DB_PATH_4_DUMP and os.path.exists(VECTOR_DB_PATH_4_DUMP):
        print(f"Loading vector store from: {VECTOR_DB_PATH_4_DUMP}")
        vector_store = InMemoryVectorStore.load(VECTOR_DB_PATH_4_DUMP, embedding=embeddings)
    else:
        print("Rebuilding vector store")
        vector_store = InMemoryVectorStore(embeddings)
        _ = await vector_store.aadd_documents(documents=documents)
        if VECTOR_DB_PATH_4_DUMP:
            print(f"Dumping vector store to: {VECTOR_DB_PATH_4_DUMP}")
            vector_store.dump(VECTOR_DB_PATH_4_DUMP)
    return vector_store


async def main() -> None:
    if POPULATE_VECTOR_DB:
        vector_store = await populate_documents()
    else:
        vector_store = None
    llm = OllamaChatModel("llama3.2:latest")
    reranker = DocumentsRerankWithLLM(llm)

    agent = RAGAgent(llm=llm, memory=UnconstrainedMemory(), vector_store=vector_store, reranker=reranker)

    response = await agent.run(RunInput(message=UserMessage("What agents are available in BeeAI?")))
    print(response.message.text)


if __name__ == "__main__":
    try:
        asyncio.run(main())
        # llm_generation()
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
