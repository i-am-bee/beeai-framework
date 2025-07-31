# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol

from beeai_framework.backend.embedding import EmbeddingModel
from beeai_framework.backend.types import Document, DocumentWithScore
from beeai_framework.backend.utils import load_module, parse_module


class QueryLike(Protocol):
    def __str__(self) -> str: ...


__all__ = ["QueryLike", "VectorStore"]


class VectorStore(ABC):
    @classmethod
    @abstractmethod
    def _class_from_name(cls, class_name: str, embedding_model: EmbeddingModel, **kwargs: Any) -> VectorStore:
        raise NotImplementedError("Implement me")

    @classmethod
    def from_name(cls, name: str, *, embedding_model: EmbeddingModel, **kwargs: Any) -> VectorStore:
        """
        Import and instantiate a VectorStore class dynamically.

        Parameters
        ----------
        name : str
            A *case sensitive* string in the format "integration:ClassName".
            - `integration` is the name of the Python package namespace (e.g. "langchain").
            - `ClassName` is the name of the vector store class to load (e.g. "Milvus").

        embedding_model : EmbeddingModel
            An instance of the embedding model required to initialize the vector store.

        **kwargs :
            any positional or keywords arguments that would be passed to the class

        Returns
        -------
        VectorStore
            An instantiated vector store object of the requested class.

        Raises
        ------
        ImportError
            If the specified class cannot be found in any known integration package.
        """
        parsed_module = parse_module(name)
        TargetVectorStore = load_module(parsed_module.provider_id, "vector_store")  # type: ignore # noqa: N806
        return TargetVectorStore._class_from_name(  # type: ignore[no-any-return]
            class_name=parsed_module.entity_id, embedding_model=embedding_model, **kwargs
        )

    @abstractmethod
    async def add_documents(self, documents: list[Document]) -> list[str]:
        raise NotImplementedError("Implement me")

    @abstractmethod
    async def search(self, query: QueryLike, k: int = 4, **kwargs: Any) -> list[DocumentWithScore]:
        raise NotImplementedError("Implement me")
