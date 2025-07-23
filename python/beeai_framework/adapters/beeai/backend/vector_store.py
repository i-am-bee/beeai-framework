# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC
from typing import Any

from beeai_framework.backend.embedding import EmbeddingModel
from beeai_framework.backend.types import Document, DocumentWithScore
from beeai_framework.backend.vector_store import QueryLike, VectorStore

try:
    from langchain_core.documents import Document as LCDocument
    from langchain_core.vectorstores import InMemoryVectorStore as LCInMemoryVectorStore

    from beeai_framework.adapters.langchain.mappers.documents import document_to_lc_document, lc_document_to_document
    from beeai_framework.adapters.langchain.mappers.embedding import LangChainBeeAIEmbeddingModel
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [rag] not found.\nRun 'pip install \"beeai-framework[rag]\"' to install."
    ) from e


class BeeAIVectorStore(VectorStore, ABC):
    @classmethod
    def _class_from_name(cls, class_name: str, embedding_model: EmbeddingModel, **kwargs: Any) -> BeeAIVectorStore:
        """Create an instance from class name (required by VectorStore base class)."""
        # Get the current module to look for classes
        import sys

        current_module = sys.modules[cls.__module__]
        # Try to get the class from the current module
        try:
            target_class = getattr(current_module, class_name)
            if not issubclass(target_class, BeeAIVectorStore):
                raise ValueError(f"Class '{class_name}' is not a BeeAIVectorStore subclass")
            instance: BeeAIVectorStore = target_class(embedding_model=embedding_model, **kwargs)
            return instance
        except AttributeError:
            raise ValueError(f"Class '{class_name}' not found for BeeAI provider")


class TemporalVectorStore(BeeAIVectorStore):
    """In-memory vector store implementation using LangChain's InMemoryVectorStore."""

    def __init__(self, embedding_model: EmbeddingModel) -> None:
        self.embedding_model = embedding_model
        self.vector_store = LCInMemoryVectorStore(embedding=LangChainBeeAIEmbeddingModel(self.embedding_model))

    async def add_documents(self, documents: list[Document]) -> list[str]:
        """Add documents to the vector store."""
        lc_documents = [document_to_lc_document(document) for document in documents]
        return await self.vector_store.aadd_documents(lc_documents)

    async def search(self, query: QueryLike, k: int = 4, **kwargs: Any) -> list[DocumentWithScore]:
        """Search for similar documents."""
        if self.vector_store is None:
            raise ValueError("Vector store must be set before searching for documents")

        query_str = str(query)
        lc_documents_with_scores: list[
            tuple[LCDocument, float]
        ] = await self.vector_store.asimilarity_search_with_score(query=query_str, k=k, **kwargs)
        documents_with_scores = [
            DocumentWithScore(document=lc_document_to_document(lc_document), score=score)
            for lc_document, score in lc_documents_with_scores
        ]
        return documents_with_scores

    def dump(self, path: str) -> None:
        """Save the vector store to disk."""
        self.vector_store.dump(path=path)

    @classmethod
    def load(cls, path: str, embedding_model: EmbeddingModel) -> TemporalVectorStore:
        """Load a vector store from disk."""
        new_vector_store = cls(embedding_model=embedding_model)
        new_vector_store.vector_store = LCInMemoryVectorStore.load(
            path=path, embedding=LangChainBeeAIEmbeddingModel(embedding_model)
        )
        return new_vector_store
