# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
from typing import Any

try:
    from langchain_core.document_loaders import BaseLoader as LCBaseLoader
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [langchain] not found.\nRun 'pip install \"beeai-framework[langchain]\"' to install."
    ) from e

from beeai_framework.adapters.langchain.mappers.documents import lc_document_to_document
from beeai_framework.backend.document_loader import DocumentLoader
from beeai_framework.backend.types import Document
from beeai_framework.logger import Logger

logger = Logger(__name__)


class LangChainDocumentLoader(DocumentLoader):
    def __init__(self, *, document_loader: LCBaseLoader) -> None:
        super().__init__()
        self.document_loader: LCBaseLoader = document_loader

    @classmethod
    def _class_from_name(cls, class_name: str, **kwargs: Any) -> LangChainDocumentLoader:
        """
        Dynamically imports and instantiates `class_name` from LangChain document loaders
        """
        lc_document_loader = None

        # Try importing from langchain_community first (most common document loaders)
        if lc_document_loader is None:
            try:
                module = importlib.import_module("langchain_community.document_loaders")
                cls_obj = getattr(module, class_name)
                lc_document_loader = cls_obj(**kwargs)
            except (ImportError, AttributeError):
                logger.info(
                    f"Failed to import LangChain document loader {class_name} from langchain_community, \
                            trying other LangChain modules"
                )

        # Try importing from langchain_core (core document loaders)
        if lc_document_loader is None:
            try:
                module = importlib.import_module("langchain_core.document_loaders")
                cls_obj = getattr(module, class_name)
                lc_document_loader = cls_obj(**kwargs)
            except (ImportError, AttributeError):
                logger.info(
                    f"Failed to import LangChain document loader {class_name} from langchain_core, \
                            trying legacy langchain module"
                )

        # Final resort: try importing from legacy langchain module
        if lc_document_loader is None:
            try:
                module = importlib.import_module("langchain.document_loaders")
                cls_obj = getattr(module, class_name)
                lc_document_loader = cls_obj(**kwargs)
            except (ImportError, AttributeError) as e:
                logger.error(f"Failed to import class {class_name}")
                raise ImportError(f"Could not import {class_name} from any LangChain document loader modules") from e

        return cls(document_loader=lc_document_loader)

    async def load(self) -> list[Document]:
        lc_documents = await self.document_loader.aload()
        return [lc_document_to_document(lc_document) for lc_document in lc_documents]
