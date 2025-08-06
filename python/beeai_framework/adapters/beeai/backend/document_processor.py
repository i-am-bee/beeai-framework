# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC
from typing import Any

from beeai_framework.adapters.llama_index.mappers.chat import LlamaIndexChatModel
from beeai_framework.adapters.llama_index.mappers.documents import (
    doc_with_score_to_li_doc_with_score,
    li_doc_with_score_to_doc_with_score,
)
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.document_processor import DocumentProcessor
from beeai_framework.backend.types import DocumentWithScore

try:
    from llama_index.core.postprocessor.llm_rerank import LLMRerank
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [rag] not found.\nRun 'pip install \"beeai-framework[rag]\"' to install."
    ) from e


class BeeAIDocumentProcessor(DocumentProcessor, ABC):
    @classmethod
    def _class_from_name(cls, class_name: str, **kwargs: Any) -> BeeAIDocumentProcessor:
        """Create an instance from class name (required by DocumentProcessor base class)."""
        # Get the current module to look for classes
        import sys

        current_module = sys.modules[cls.__module__]
        # Try to get the class from the current module
        try:
            target_class = getattr(current_module, class_name)
            if not issubclass(target_class, DocumentProcessor):
                raise ValueError(f"Class '{class_name}' is not a DocumentProcessor subclass")
            instance: BeeAIDocumentProcessor = target_class(**kwargs)
            return instance
        except AttributeError:
            raise ValueError(f"Class '{class_name}' not found for BeeAI provider")


class LLMDocumentReranker(BeeAIDocumentProcessor):
    def __init__(self, llm: ChatModel, *, choice_batch_size: int = 5, top_n: int = 5) -> None:
        self.llm = llm
        self.reranker = LLMRerank(
            choice_batch_size=choice_batch_size, top_n=top_n, llm=LlamaIndexChatModel(llm=self.llm)
        )

    async def postprocess_documents(
        self, documents: list[DocumentWithScore], *, query: str | None = None
    ) -> list[DocumentWithScore]:
        if query is None:
            raise ValueError("DocumentsRerankWithLLM requires 'query' parameter for reranking")

        li_documents_with_score = [doc_with_score_to_li_doc_with_score(document) for document in documents]
        processed_nodes = await self.reranker.apostprocess_nodes(li_documents_with_score, query_str=query)
        documents_with_score = [li_doc_with_score_to_doc_with_score(node) for node in processed_nodes]
        return documents_with_score
