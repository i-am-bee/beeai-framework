import asyncio
import sys
import traceback

from beeai_framework.adapters.beeai.backend.vector_store import TemporalVectorStore
from beeai_framework.adapters.langchain.mappers.documents import lc_document_to_document
from beeai_framework.backend.embedding import EmbeddingModel
from beeai_framework.backend.vector_store import VectorStore
from beeai_framework.errors import FrameworkError

# LC dependencies - to be swapped with BAI dependencies
try:
    from langchain_community.document_loaders import UnstructuredMarkdownLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional modules are not found.\nRun 'pip install \"beeai-framework[rag]\"' to install."
    ) from e


async def main() -> None:
    embedding_model = EmbeddingModel.from_name("watsonx:ibm/slate-125m-english-rtrvr-v2", truncate_input_tokens=500)

    # Document loading
    loader = UnstructuredMarkdownLoader(file_path="docs/modules/agents.mdx")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=1000)
    all_splits = text_splitter.split_documents(docs)
    documents = [lc_document_to_document(document) for document in all_splits]
    print(f"Loaded {len(documents)} documents")

    vector_store: TemporalVectorStore = VectorStore.from_name(
        name="beeai:TemporalVectorStore", embedding_model=embedding_model
    )  # type: ignore[assignment]
    _ = await vector_store.add_documents(documents=documents)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
