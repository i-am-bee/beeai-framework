# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0
import uuid
from abc import ABC
from typing import Any

from beeai_framework.backend import EmbeddingModelOutput
from beeai_framework.backend.embedding import EmbeddingModel
from beeai_framework.backend.types import Document, DocumentWithScore
from beeai_framework.backend.vector_store import QueryLike, VectorStore

try:
    from agentstack_sdk.platform import VectorStore as AgentStackSDKVectorStore
    from agentstack_sdk.platform import VectorStoreItem
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [agentstack] not found.\nRun 'pip install \"beeai-framework[agentstack]\"' to install."
    ) from e


class AgentStackVectorStore(VectorStore, ABC):
    @classmethod
    def _class_from_name(cls, class_name: str, embedding_model: EmbeddingModel, **kwargs: Any) -> VectorStore:
        """Create an instance from class name (required by VectorStore base class)."""
        # Get the current module to look for classes
        import sys

        current_module = sys.modules[cls.__module__]
        # Try to get the class from the current module
        try:
            target_class = getattr(current_module, class_name)
            if not issubclass(target_class, NativeVectorStore):
                raise ValueError(f"Class '{class_name}' is not a NativeVectorStore subclass")
            instance = target_class(embedding_model=embedding_model, **kwargs)
            return instance
        except AttributeError:
            raise ValueError(f"Class '{class_name}' not found for BeeAI provider")


class NativeVectorStore(AgentStackVectorStore):
    def __init__(self, embedding_model: EmbeddingModel, *, name: str | None = None) -> None:
        self.embedding_model = embedding_model
        self._vector_store = None
        self._name = name or f"rag-{uuid.uuid4()}"

    async def add_documents(self, documents: list[Document]) -> list[str]:
        """Add documents to the vector store."""
        vector_store = await self._get_vector_store()
        embedding_response: EmbeddingModelOutput = await self.embedding_model.create(
            values=[document.content for document in documents]
        )
        as_documents = [
            VectorStoreItem(
                document_type="external",
                text=embedding_response.values[index],
                embedding=embedding,  # pyrefly: ignore [bad-argument-type]
                document_id=str(index),
            )
            for index, embedding in enumerate(embedding_response.embeddings)
        ]
        await vector_store.add_documents(as_documents)
        return embedding_response.values

    async def search(self, query: QueryLike, k: int = 4, **kwargs: Any) -> list[DocumentWithScore]:
        """Search for similar documents."""
        vector_store = await self._get_vector_store()

        query_str = str(query)
        embedding_response: EmbeddingModelOutput = await self.embedding_model.create(values=[query_str])
        as_documents_with_scores = await vector_store.search(embedding_response.embeddings[0], limit=k, **kwargs)
        documents_with_scores = [
            DocumentWithScore(
                document=Document(
                    content=as_document.item.text,
                    metadata=as_document.item.metadata or {},  # pyrefly: ignore [bad-argument-type]
                ),
                score=as_document.score,
            )
            for as_document in as_documents_with_scores
        ]
        return documents_with_scores

    async def _get_vector_store(self) -> AgentStackSDKVectorStore:
        if self._vector_store is None:
            embedding_response: EmbeddingModelOutput = await self.embedding_model.create(values=["test"])
            dimension = len(embedding_response.embeddings[0])
            self._vector_store = await AgentStackSDKVectorStore.create(
                name=self._name, dimension=dimension, model_id=self.embedding_model.model_id
            )
        return self._vector_store

    @property
    def is_initialized(self) -> bool:
        """Check if the vector store has been initialized."""
        return self._vector_store is not None
