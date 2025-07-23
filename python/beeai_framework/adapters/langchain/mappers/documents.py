# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

try:
    from langchain_core.documents import Document
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [langchain] not found.\nRun 'pip install \"beeai-framework[langchain]\"' to install."
    ) from e


from beeai_framework.backend.types import Document as VectorStoreDocument


def lc_document_to_document(lc_document: Document) -> VectorStoreDocument:
    return VectorStoreDocument(content=lc_document.page_content, metadata=lc_document.metadata)


def document_to_lc_document(document: VectorStoreDocument) -> Document:
    return Document(page_content=document.content, metadata=document.metadata)
