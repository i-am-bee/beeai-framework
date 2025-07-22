# Copyright 2025 � BeeAI a Series of LF Projects, LLC
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

from __future__ import annotations

from typing import Any

from beeai_framework.backend.embedding import EmbeddingModel
from beeai_framework.backend.types import Document, DocumentWithScore
from beeai_framework.backend.vector_store import QueryLike, VectorStore

try:
    from langchain_core.documents import Document as LCDocument
    from langchain_core.vectorstores import InMemoryVectorStore as LCInMemoryVectorStore
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [langchain] not found.\nRun 'poetry install --extras langchain' to install."
    ) from e

from beeai_framework.adapters.langchain.mappers.documents import document_to_lc_document, lc_document_to_document
from beeai_framework.adapters.langchain.mappers.embedding import LangChainBeeAIEmbeddingModel


class InMemoryVectorStore(VectorStore):
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

    def search_sync(self, query: QueryLike, k: int = 4, **kwargs: Any) -> list[DocumentWithScore]:
        """Synchronous search for similar documents."""
        if self.vector_store is None:
            raise ValueError("Vector store must be set before searching for documents")

        query_str = str(query)
        lc_documents_with_scores: list[tuple[LCDocument, float]] = self.vector_store.similarity_search_with_score(
            query=query_str, k=k, **kwargs
        )
        documents_with_scores = [
            DocumentWithScore(document=lc_document_to_document(lc_document), score=score)
            for lc_document, score in lc_documents_with_scores
        ]
        return documents_with_scores

    def dump(self, path: str) -> None:
        """Save the vector store to disk."""
        self.vector_store.dump(path=path)

    @classmethod
    def load(cls, path: str, embedding_model: EmbeddingModel) -> InMemoryVectorStore:
        """Load a vector store from disk."""
        new_vector_store = cls(embedding_model=embedding_model)
        new_vector_store.vector_store = LCInMemoryVectorStore.load(
            path=path, embedding=LangChainBeeAIEmbeddingModel(embedding_model)
        )
        return new_vector_store

    @classmethod
    def _class_from_name(cls, class_name: str, embedding_model: EmbeddingModel, **kwargs: Any) -> InMemoryVectorStore:
        """Create an instance from class name (required by VectorStore base class)."""
        if class_name != "InMemoryVectorStore":
            raise ValueError(f"Class name must be 'InMemoryVectorStore', got '{class_name}'")
        return cls(embedding_model=embedding_model, **kwargs)
