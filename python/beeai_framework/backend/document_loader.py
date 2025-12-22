# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from beeai_framework.backend.types import Document
from beeai_framework.backend.utils import load_module, parse_module

__all__ = ["DocumentLoader"]


class DocumentLoader(ABC):
    """Abstract base class for loading documents from various sources.

    DocumentLoader provides a unified interface for loading documents from different
    sources such as files, URLs, databases, or APIs. It supports dynamic instantiation
    of provider-specific loaders through the `from_name` factory method.

    Subclasses must implement the `load` method to define how documents are loaded
    from their specific source, and the `_class_from_name` method for dynamic loading.

    Example:
        >>> # Load documents using LangChain's UnstructuredMarkdownLoader
        >>> loader = DocumentLoader.from_name(
        ...     "langchain:UnstructuredMarkdownLoader",
        ...     file_path="README.md"
        ... )
        >>> documents = await loader.load()
        >>> print(len(documents))
        >>>
        >>> # Load PDF documents
        >>> loader = DocumentLoader.from_name(
        ...     "langchain:PyPDFLoader",
        ...     file_path="document.pdf"
        ... )
        >>> documents = await loader.load()
    """

    @classmethod
    @abstractmethod
    def _class_from_name(cls, class_name: str, **kwargs: Any) -> DocumentLoader:
        """Create a document loader instance from a class name (internal method).

        This method must be implemented by integration-specific subclasses to
        handle the dynamic instantiation of document loader classes.

        Args:
            class_name: The name of the document loader class to instantiate.
            **kwargs: Arguments to pass to the document loader constructor.

        Returns:
            An instantiated DocumentLoader of the specified class.

        Raises:
            NotImplementedError: If the subclass doesn't implement this method.
        """
        raise NotImplementedError("Implement me")

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> DocumentLoader:
        """
        Import and instantiate a DocumentLoader class dynamically.

        Parameters
        ----------
        name : str
            A *case sensitive* string in the format "integration:ClassName".
            - `integration` is the name of the Python package namespace (e.g. "langchain").
            - `ClassName` is the name of the document loader class to load (e.g. "UnstructuredMarkdownLoader").

        **kwargs :
            any positional or keywords arguments that would be passed to the class

        Returns
        -------
        DocumentLoader
            An instantiated document loader object of the requested class.

        Raises
        ------
        ImportError
            If the specified class cannot be found in any known integration package.
        """
        parsed_module = parse_module(name)
        TargetDocumentLoader = load_module(parsed_module.provider_id, "document_loader")  # type: ignore # noqa: N806
        return TargetDocumentLoader._class_from_name(  # type: ignore[no-any-return]
            class_name=parsed_module.entity_id, **kwargs
        )

    @abstractmethod
    async def load(self) -> list[Document]:
        """
        Load documents from the source.

        Returns
        -------
        list[Document]
            A list of loaded documents.
        """
        raise NotImplementedError("Implement me")
