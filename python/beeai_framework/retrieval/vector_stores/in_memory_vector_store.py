from __future__ import annotations

from typing import Any

try:
    from langchain_core.documents import Document as LCDocument
    from langchain_core.vectorstores import InMemoryVectorStore as LCInMemoryVectorStore

except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [langchain] not found.\nRun 'pip install \"beeai-framework[langchain]\"' to install."
    ) from e

from beeai_framework.adapters.langchain.mappers.documents import lc_document_to_document
from beeai_framework.adapters.langchain.mappers.lc_embedding import LangChainBeeAIEmbeddingModel
from beeai_framework.backend.embedding import EmbeddingModel
from beeai_framework.backend.types import DocumentWithScore


class InMemoryVectorStore:
    def __init__(self, embedding_model: EmbeddingModel) -> None:
        super().__init__(embedding_model=embedding_model)
        self.vector_store = LCInMemoryVectorStore(embedding=LangChainBeeAIEmbeddingModel(self.embedding_model))

    async def search(self, query: str, k: int = 4, **kwargs: Any) -> list[DocumentWithScore]:
        if self.vector_store is None:
            raise ValueError("Vector store must be set before searching for documents")
        lc_documents_with_scores: list[
            tuple[LCDocument, float]
        ] = await self.vector_store.asimilarity_search_with_score(query=query, k=k, **kwargs)
        documents_with_scores = [
            DocumentWithScore(document=lc_document_to_document(lc_document), score=score)
            for lc_document, score in lc_documents_with_scores
        ]
        return documents_with_scores

    def dump(self, path: str) -> None:
        self.vector_store.dump(path=path)

    @classmethod
    def load(cls, path: str, embedding: EmbeddingModel) -> InMemoryVectorStore:
        new_vector_store = cls(embedding_model=embedding)
        new_vector_store.vector_store = LCInMemoryVectorStore.load(
            path=path, embedding=LangChainBeeAIEmbeddingModel(new_vector_store.embedding_model)
        )
        return new_vector_store
