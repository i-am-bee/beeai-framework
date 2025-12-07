# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from beeai_framework.backend.types import Document
from beeai_framework.backend.utils import load_module, parse_module

__all__ = ["TextSplitter"]


class TextSplitter(ABC):
    """Abstract base class for splitting text and documents into smaller chunks.

    TextSplitter provides a unified interface for breaking down large texts or documents
    into smaller, manageable chunks. This is essential for processing long documents that
    exceed model context limits or for creating more focused embeddings.

    The class supports dynamic instantiation of provider-specific splitters through
    the `from_name` factory method, allowing integration with various text splitting
    implementations (e.g., LangChain's RecursiveCharacterTextSplitter).

    Subclasses must implement:
        - `split_documents`: Split a list of documents into chunks
        - `split_text`: Split raw text into chunks
        - `_class_from_name`: Factory method for dynamic instantiation

    Example:
        >>> # Split documents using LangChain's RecursiveCharacterTextSplitter
        >>> splitter = TextSplitter.from_name(
        ...     "langchain:RecursiveCharacterTextSplitter",
        ...     chunk_size=1000,
        ...     chunk_overlap=200
        ... )
        >>>
        >>> # Split a list of documents
        >>> documents = [Document(content="Long text here...")]
        >>> chunks = await splitter.split_documents(documents)
        >>> print(len(chunks))
        >>>
        >>> # Split raw text
        >>> text = "Very long text that needs to be split..."
        >>> text_chunks = await splitter.split_text(text)
        >>>
        >>> # Use with different strategies
        >>> splitter = TextSplitter.from_name(
        ...     "langchain:CharacterTextSplitter",
        ...     separator="\\n\\n",
        ...     chunk_size=500
        ... )
    """

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> TextSplitter:
        """
        Import and instantiate a TextSplitter class dynamically.

        Args:
            name:
                A *case-sensitive* string in the format "integration:ClassName".
                - `integration` is the name of the Python package namespace (e.g. "langchain").
                - `ClassName` is the name of the text splitter class to load (e.g. "RecursiveCharacterTextSplitter").
            **kwargs:
                Additional positional or keyword arguments to be passed to the class.

        Returns:
            TextSplitter:
                An instantiated text splitter object of the requested class.

        Raises:
            ImportError:
                If the specified class cannot be found in any known integration package.
            ValueError:
                If the provided name is not in the required "integration:ClassName" format.
        """
        parsed_module = parse_module(name)
        if not parsed_module.entity_id:
            raise ValueError(
                f"Only provider {parsed_module.provider_id} was specified. Text Splitter name was not specified."
            )

        target: type[TextSplitter] = load_module(parsed_module.provider_id, "text_splitter")
        return target._class_from_name(class_name=parsed_module.entity_id, **kwargs)

    @abstractmethod
    async def split_documents(self, documents: list[Document]) -> list[Document]:
        """Split a list of documents into smaller chunks.

        This method takes a list of documents and splits each one into smaller
        chunks based on the splitter's configuration (chunk size, overlap, etc.).
        Each chunk becomes a new Document with the same metadata as the original.

        Args:
            documents: List of documents to split into chunks.

        Returns:
            A list of Document objects, where each represents a chunk of the
            original documents.

        Raises:
            NotImplementedError: If the subclass doesn't implement this method.

        Example:
            >>> documents = [
            ...     Document(content="Long document text...", metadata={"source": "file.txt"})
            ... ]
            >>> chunks = await splitter.split_documents(documents)
            >>> print(len(chunks))  # More chunks than original documents
        """
        raise NotImplementedError("Implement me")

    @abstractmethod
    async def split_text(self, text: str) -> list[str]:
        """Split text into smaller chunks.

        This method takes a raw text string and splits it into smaller chunks
        based on the splitter's configuration. Unlike `split_documents`, this
        returns plain strings without metadata.

        Args:
            text: The text string to split into chunks.

        Returns:
            A list of text strings, where each represents a chunk of the
            original text.

        Raises:
            NotImplementedError: If the subclass doesn't implement this method.

        Example:
            >>> text = "This is a very long text that needs to be split..."
            >>> chunks = await splitter.split_text(text)
            >>> print(len(chunks))
            >>> print(chunks[0])  # First chunk
        """
        raise NotImplementedError("Implement me")

    @classmethod
    @abstractmethod
    def _class_from_name(cls, class_name: str, **kwargs: Any) -> TextSplitter:
        """Create a text splitter instance from a class name (internal method).

        This method must be implemented by integration-specific subclasses to
        handle the dynamic instantiation of text splitter classes.

        Args:
            class_name: The name of the text splitter class to instantiate.
            **kwargs: Arguments to pass to the text splitter constructor.

        Returns:
            An instantiated TextSplitter of the specified class.

        Raises:
            NotImplementedError: If the subclass doesn't implement this method.
        """
        raise NotImplementedError("Implement me")
