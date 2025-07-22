# Copyright 2025 © BeeAI a Series of LF Projects, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import logging
import os
import sys
import traceback

from beeai_framework.adapters.langchain.backend.vector_store import LangChainVectorStore
from beeai_framework.adapters.langchain.mappers.documents import lc_document_to_document
from beeai_framework.adapters.langchain.mappers.lc_embedding import LangChainBeeAIEmbeddingModel
from beeai_framework.adapters.watsonx.backend.embedding import WatsonxEmbeddingModel
from beeai_framework.agents.experimental import RAGAgent, RagAgentRunInput
from beeai_framework.backend import UserMessage
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.vector_store import VectorStore
from beeai_framework.errors import FrameworkError
from beeai_framework.logger import Logger
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.retrieval.document_processors.document_processors import DocumentsRerankWithLLM

# LC dependencies - to be swapped with BAI dependencies
try:
    from langchain_community.document_loaders import UnstructuredMarkdownLoader
    from langchain_core.vectorstores import InMemoryVectorStore as LCInMemoryVectorStore
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional modules are not found.\nRun 'poetry install --extras langchain' to install."
    ) from e


from dotenv import load_dotenv

load_dotenv()  # load environment variables
logger = Logger("rag-agent", level=logging.DEBUG)


POPULATE_VECTOR_DB = True
VECTOR_DB_PATH_4_DUMP = ""  # Set this path for persistency
INPUT_DOCUMENTS_LOCATION = "docs-mintlify/integrations"


async def populate_documents() -> VectorStore | None:
    embedding_model = WatsonxEmbeddingModel(  # type: ignore[call-arg]
        model_id="ibm/slate-125m-english-rtrvr-v2",
        project_id=os.getenv("WATSONX_PROJECT_ID"),
        api_key=os.getenv("WATSONX_APIKEY"),
        base_url=os.getenv("WATSONX_URL"),
        truncate_input_tokens=500,  # Base class parameter
    )

    # Load existing vector store if available
    if VECTOR_DB_PATH_4_DUMP and os.path.exists(VECTOR_DB_PATH_4_DUMP):
        print(f"Loading vector store from: {VECTOR_DB_PATH_4_DUMP}")
        lc_embedding = LangChainBeeAIEmbeddingModel(embedding_model)
        lc_inmemory_vector_store = LCInMemoryVectorStore.load(path=VECTOR_DB_PATH_4_DUMP, embedding=lc_embedding)
        vector_store = LangChainVectorStore(vector_store=lc_inmemory_vector_store)
        return vector_store
    
    # Create new vector store if population is enabled
    if POPULATE_VECTOR_DB:
        loader = UnstructuredMarkdownLoader(file_path="python/docs/agents.md")
        try:
            docs = loader.load()
        except Exception:
            return None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=1000)
        all_splits = text_splitter.split_documents(docs)
        documents = [lc_document_to_document(document) for document in all_splits]
        print(f"Loaded {len(documents)} documents")

        print("Rebuilding vector store")
        vector_store = VectorStore.from_name(name="langchain:InMemoryVectorStore", embedding_model=embedding_model)  # type: ignore[assignment]
        # vector_store = InMemoryVectorStore(embedding_model)
        _ = await vector_store.add_documents(documents=documents)
        if VECTOR_DB_PATH_4_DUMP and isinstance(vector_store, LangChainVectorStore):
            print(f"Dumping vector store to: {VECTOR_DB_PATH_4_DUMP}")
            vector_store.vector_store.dump(VECTOR_DB_PATH_4_DUMP)  # type: ignore[attr-defined]
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
    
    llm = ChatModel.from_name("ollama:llama3.2:latest")
    reranker = DocumentsRerankWithLLM(llm)

    agent = RAGAgent(llm=llm, memory=UnconstrainedMemory(), vector_store=vector_store, reranker=reranker)

    response = await agent.run(RagAgentRunInput(message=UserMessage("What agents are available in BeeAI?")))
    print(response.message.text)


if __name__ == "__main__":
    try:
        asyncio.run(main())
        # llm_generation()
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
