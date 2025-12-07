# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol

from beeai_framework.backend.embedding import EmbeddingModel
from beeai_framework.backend.types import Document, DocumentWithScore
from beeai_framework.backend.utils import load_module, parse_module


class QueryLike(Protocol):
    """Protocol for objects that can be converted to query strings.

    Any object that implements the `__str__` method can be used as a query
    for vector store searches.
    """
    def __str__(self) -> str: ...


__all__ = ["QueryLike", "VectorStore"]


class VectorStore(ABC):
    """Abstract base class for vector database implementations.

    VectorStore provides a unified interface for storing and retrieving document
    embeddings in vector databases. It supports semantic search by finding documents
    similar to a query based on vector similarity.

    The class handles:
        - Adding documents with automatic embedding generation
        - Semantic search with similarity scoring
        - Dynamic instantiation of provider-specific stores

    Subclasses must implement:
        - `add_documents`: Store documents in the vector database
        - `search`: Perform similarity search
        - `_class_from_name`: Factory method for dynamic instantiation

    Example:
        >>> from beeai_framework.backend import EmbeddingModel, VectorStore
        >>>
        >>> # Create embedding model
        >>> embedding_model = EmbeddingModel.from_name("openai:text-embedding-3-small")
        >>>
        >>> # Create vector store
        >>> vector_store = VectorStore.from_name(
        ...     "langchain:Chroma",
        ...     embedding_model=embedding_model,
        ...     collection_name="my_documents"
        ... )
        >>>
        >>> # Add documents
        >>> documents = [
        ...     Document(content="Python is a programming language", metadata={"source": "doc1"}),
        ...     Document(content="JavaScript is used for web development", metadata={"source": "doc2"})
        ... ]
        >>> ids = await vector_store.add_documents(documents)
        >>>
        >>> # Search for similar documents
        >>> results = await vector_store.search("programming languages", k=2)
        >>> for doc_with_score in results:
        ...     print(f"Score: {doc_with_score.score}, Content: {doc_with_score.document.content}")
        >>>
        >>> # Use different vector stores
        >>> vector_store = VectorStore.from_name(
        ...     "langchain:FAISS",
        ...     embedding_model=embedding_model
        ... )
    """

    @classmethod
    def from_name(cls, name: str, *, embedding_model: EmbeddingModel, **kwargs: Any) -> VectorStore:
        """
        Import and instantiate a VectorStore class dynamically.

        Args:
            name:
                A *case-sensitive* string in the format "integration:ClassName".
                - `integration` is the name of the Python package namespace (e.g. "langchain").
                - `ClassName` is the name of the vector store class to load (e.g. "Milvus").
            embedding_model:
                An instance of the embedding model required to initialize the vector store.
            **kwargs:
                Additional positional or keyword arguments to be passed to the class.

        Returns:
            VectorStore:
                An instantiated vector store object of the requested class.

        Raises:
            ImportError:
                If the specified class cannot be found in any known integration package.
            ValueError:
                If the provided name is not in the required "integration:ClassName" format.
        """
        parsed_module = parse_module(name)
        if not parsed_module.entity_id:
            raise ValueError(
                f"Only provider {parsed_module.provider_id} was specified. Vector Store name was not specified."
            )

        target: type[VectorStore] = load_module(parsed_module.provider_id, "vector_store")
        return target._class_from_name(
            class_name=parsed_module.entity_id,
            embedding_model=embedding_model,
            **kwargs,
        )

    @abstractmethod
    async def add_documents(self, documents: list[Document]) -> list[str]:
        """Add documents to the vector store.

        This method takes a list of documents, generates embeddings for their content
        using the configured embedding model, and stores them in the vector database.
        Each document is assigned a unique identifier.

        Args:
            documents: List of Document objects to add to the vector store.
                Each document should have content and optional metadata.

        Returns:
            A list of unique identifiers (IDs) for the added documents,
            in the same order as the input documents.

        Raises:
            NotImplementedError: If the subclass doesn't implement this method.

        Example:
            >>> documents = [
            ...     Document(content="First document", metadata={"source": "file1.txt"}),
            ...     Document(content="Second document", metadata={"source": "file2.txt"})
            ... ]
            >>> ids = await vector_store.add_documents(documents)
            >>> print(ids)  # ['id1', 'id2']
        """
        raise NotImplementedError("Implement me")

    @abstractmethod
    async def search(self, query: QueryLike, k: int = 4, **kwargs: Any) -> list[DocumentWithScore]:
        """Search for documents similar to the query.

        This method performs semantic search by converting the query to an embedding
        and finding the k most similar documents in the vector store based on
        vector similarity (e.g., cosine similarity).

        Args:
            query: The search query. Can be a string or any object implementing
                the QueryLike protocol (has __str__ method).
            k: The number of most similar documents to return. Defaults to 4.
            **kwargs: Additional provider-specific search parameters
                (e.g., filter conditions, score threshold).

        Returns:
            A list of DocumentWithScore objects, sorted by similarity score
            (highest first). Each contains the document and its similarity score.

        Raises:
            NotImplementedError: If the subclass doesn't implement this method.

        Example:
            >>> # Basic search
            >>> results = await vector_store.search("machine learning", k=3)
            >>> for result in results:
            ...     print(f"Score: {result.score:.3f}")
            ...     print(f"Content: {result.document.content}")
            >>>
            >>> # Search with filters (provider-specific)
            >>> results = await vector_store.search(
            ...     "python programming",
            ...     k=5,
            ...     filter={"source": "documentation"}
            ... )
        """
        raise NotImplementedError("Implement me")

    @classmethod
    @abstractmethod
    def _class_from_name(cls, class_name: str, embedding_model: EmbeddingModel, **kwargs: Any) -> VectorStore:
        """Create a vector store instance from a class name (internal method).

        This method must be implemented by integration-specific subclasses to
        handle the dynamic instantiation of vector store classes.

        Args:
            class_name: The name of the vector store class to instantiate.
            embedding_model: The embedding model to use for generating vectors.
            **kwargs: Arguments to pass to the vector store constructor.

        Returns:
            An instantiated VectorStore of the specified class.

        Raises:
            NotImplementedError: If the subclass doesn't implement this method.
        """
        raise NotImplementedError("Implement me")
