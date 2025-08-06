# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from beeai_framework.backend.types import Document
from beeai_framework.backend.utils import load_module, parse_module

__all__ = ["TextSplitter"]


class TextSplitter(ABC):
    @classmethod
    @abstractmethod
    def _class_from_name(cls, class_name: str, **kwargs: Any) -> TextSplitter:
        raise NotImplementedError("Implement me")

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> TextSplitter:
        """
        Import and instantiate a TextSplitter class dynamically.

        Parameters
        ----------
        name : str
            A *case sensitive* string in the format "integration:ClassName".
            - `integration` is the name of the Python package namespace (e.g. "langchain").
            - `ClassName` is the name of the text splitter class to load (e.g. "RecursiveCharacterTextSplitter").

        **kwargs :
            any positional or keywords arguments that would be passed to the class

        Returns
        -------
        TextSplitter
            An instantiated text splitter object of the requested class.

        Raises
        ------
        ImportError
            If the specified class cannot be found in any known integration package.
        """
        parsed_module = parse_module(name)
        TargetTextSplitter = load_module(parsed_module.provider_id, "text_splitter")  # type: ignore # noqa: N806
        return TargetTextSplitter._class_from_name(  # type: ignore[no-any-return]
            class_name=parsed_module.entity_id, **kwargs
        )

    @abstractmethod
    async def split_documents(self, documents: list[Document]) -> list[Document]:
        """
        Split a list of documents into smaller chunks.

        Parameters
        ----------
        documents : list[Document]
            The documents to split.

        Returns
        -------
        list[Document]
            A list of document chunks.
        """
        raise NotImplementedError("Implement me")

    @abstractmethod
    async def split_text(self, text: str) -> list[str]:
        """
        Split text into smaller chunks.

        Parameters
        ----------
        text : str
            The text to split.

        Returns
        -------
        list[str]
            A list of text chunks.
        """
        raise NotImplementedError("Implement me")
