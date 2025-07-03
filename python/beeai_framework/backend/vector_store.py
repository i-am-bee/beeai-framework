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

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from beeai_framework.backend.embedding import EmbeddingModel
from beeai_framework.backend.types import Document, DocumentWithScore


class VectorStore(ABC):
    _integration_registry: ClassVar[dict[str, type[VectorStore]]] = {}

    def __init_subclass__(cls, /, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._integration_registry[cls.integration_name.lower()] = cls

    @abstractmethod
    def _class_from_name(self, class_name: str, embedding_model: EmbeddingModel, **kwargs: Any) -> VectorStore:
        # Every implemen
        raise NotImplementedError("Implement me")

    @classmethod
    def from_name(cls, name: str, embedding_model: EmbeddingModel, **kwargs: Any) -> VectorStore:
        """
        Import and instantiate a VectorStore class dynamically.

        Parameters
        ----------
        name : str
            A *case sensitive* string in the format "integration/ClassName".
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
        integration_name = name[: name.find("/")].lower()
        if integration_name not in cls._integration_registry:
            raise ImportError(f"Unknown integration: {integration_name}")
        return cls._integration_registry[integration_name]._class_from_name(
            class_name=name[name.find("/") + 1 :], embedding_model=embedding_model, **kwargs
        )

    @abstractmethod
    async def add_documents(self, documents: Document) -> list[str]:
        raise NotImplementedError("Implement me")

    @abstractmethod
    async def search(self, query: str, search_type: str, k: int = 4, **kwargs: Any) -> list[DocumentWithScore]:
        raise NotImplementedError("Implement me")
