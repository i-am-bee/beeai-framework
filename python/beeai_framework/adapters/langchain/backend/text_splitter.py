# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
from typing import Any

try:
    from langchain_text_splitters import TextSplitter as LCTextSplitter
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [langchain] not found.\nRun 'pip install \"beeai-framework[langchain]\"' to install."
    ) from e

from beeai_framework.adapters.langchain.mappers.documents import document_to_lc_document, lc_document_to_document
from beeai_framework.backend.text_splitter import TextSplitter
from beeai_framework.backend.types import Document
from beeai_framework.logger import Logger

logger = Logger(__name__)


class LangChainTextSplitter(TextSplitter):
    def __init__(self, *, text_splitter: LCTextSplitter) -> None:
        super().__init__()
        self.text_splitter: LCTextSplitter = text_splitter

    @classmethod
    def _class_from_name(cls, class_name: str, **kwargs: Any) -> LangChainTextSplitter:
        """
        Dynamically imports and instantiates `class_name` from LangChain text splitters
        """
        lc_text_splitter = None

        # Try importing from langchain_text_splitters first (most common text splitters)
        try:
            module = importlib.import_module("langchain_text_splitters")
            cls_obj = getattr(module, class_name)
            lc_text_splitter = cls_obj(**kwargs)
        except (ImportError, AttributeError):
            logger.info(
                f"Failed to import LangChain text splitter {class_name} from langchain_text_splitters, \
                        trying other LangChain modules"
            )

        # Final resort: try importing from legacy langchain module
        if lc_text_splitter is None:
            try:
                module = importlib.import_module("langchain.text_splitter")
                cls_obj = getattr(module, class_name)
                lc_text_splitter = cls_obj(**kwargs)
            except (ImportError, AttributeError) as e:
                logger.error(f"Failed to import class {class_name}")
                raise ImportError(f"Could not import {class_name} from any LangChain text splitter modules") from e

        return cls(text_splitter=lc_text_splitter)

    async def split_documents(self, documents: list[Document]) -> list[Document]:
        lc_documents = [document_to_lc_document(doc) for doc in documents]
        split_lc_documents = self.text_splitter.split_documents(lc_documents)
        return [lc_document_to_document(lc_doc) for lc_doc in split_lc_documents]

    async def split_text(self, text: str) -> list[str]:
        return self.text_splitter.split_text(text)
