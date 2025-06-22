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

from __future__ import annotations

from typing import Any

from beeai_framework.backend.types import Document, DocumentWithScore

try:
    from langchain_core.documents import Document as LCDocument
    from langchain_core.vectorstores import InMemoryVectorStore as LCInMemoryVectorStore
    from langchain_core.vectorstores import VectorStore as VectorStoreFromLC

except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [langchain] not found.\nRun 'pip install \"beeai-framework[langchain]\"' to install."
    ) from e

from beeai_framework.adapters.langchain.utils import document_to_lc_document, lc_document_to_document
from beeai_framework.adapters.langchain.wrappers.lc_embedding import LCEmbedding
from beeai_framework.backend.embedding import EmbeddingModel
from beeai_framework.backend.vector_store import VectorStore


class LCVectorStore(VectorStore):
    def __init__(self, embedding_model: EmbeddingModel) -> None:
        super().__init__(embedding_model=embedding_model)
        self.vector_store: VectorStoreFromLC = None

    @classmethod
    def instantiate_by_name(cls, vector_store_name: str, **kwards: dict[Any, Any]) -> LCVectorStore:
        raise NotImplementedError("TBD")

    def add_documents(self, documents: Document) -> list[str]:
        """Add or update documents in the vectorstore, similar to the behavior of LangChain's `add_documnets`.

        Args:
            documents: documents to add to the vectorstore.

        Returns:
            List of IDs of the added documents.
        """
        if self.vector_store is None:
            raise ValueError("Vector store must be set before adding documents")
        lc_documents = [document_to_lc_document(document) for document in documents]
        return self.vector_store.add_documents(lc_documents)

    async def aadd_documents(self, documents: Document) -> list[str]:
        if self.vector_store is None:
            raise ValueError("Vector store must be set before adding documents")
        lc_documents = [document_to_lc_document(document) for document in documents]
        return await self.vector_store.aadd_documents(lc_documents)

    def search(self, query: str, k: int = 4, **kwargs: Any) -> list[DocumentWithScore]:
        if self.vector_store is None:
            raise ValueError("Vector store must be set before searching for documents")
        lc_documents_with_scores: list[tuple[LCDocument, float]] = (
            self.vector_store.similarity_search_with_relevance_scores(query=query, k=k, **kwargs)
        )
        documents_with_scores = [
            DocumentWithScore(document=lc_document_to_document(lc_document), score=score)
            for lc_document, score in lc_documents_with_scores
        ]
        return documents_with_scores

    async def asearch(self, query: str, k: int = 4, **kwargs: Any) -> list[DocumentWithScore]:
        if self.vector_store is None:
            raise ValueError("Vector store must be set before searching for documents")
        lc_documents_with_scores: list[
            tuple[LCDocument, float]
        ] = await self.vector_store.asimilarity_search_with_relevance_scores(query=query, k=k, **kwargs)
        documents_with_scores = [
            DocumentWithScore(document=lc_document_to_document(lc_document), score=score)
            for lc_document, score in lc_documents_with_scores
        ]
        return documents_with_scores


class InMemoryVectorStore(LCVectorStore):
    def __init__(self, embedding_model: EmbeddingModel) -> None:
        super().__init__(embedding_model=embedding_model)
        self.vector_store = LCInMemoryVectorStore(embedding=LCEmbedding(embedding=self.embedding_model))

    def search(self, query: str, k: int = 4, **kwargs: Any) -> list[DocumentWithScore]:
        if self.vector_store is None:
            raise ValueError("Vector store must be set before searching for documents")
        lc_documents_with_scores: list[tuple[LCDocument, float]] = self.vector_store.similarity_search_with_score(
            query=query, k=k, **kwargs
        )
        documents_with_scores = [
            DocumentWithScore(document=lc_document_to_document(lc_document), score=score)
            for lc_document, score in lc_documents_with_scores
        ]
        return documents_with_scores

    async def asearch(self, query: str, k: int = 4, **kwargs: Any) -> list[DocumentWithScore]:
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
            path=path, embedding=LCEmbedding(new_vector_store.embedding_model)
        )
        return new_vector_store
