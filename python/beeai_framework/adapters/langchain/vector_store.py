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

import importlib
from typing import Any

from beeai_framework.adapters.langchain.mappers.lc_embedding import LangChainBeeAIEmbeddingModel
from beeai_framework.backend.types import Document, DocumentWithScore

try:
    from langchain_core.documents import Document as LCDocument
    from langchain_core.vectorstores import VectorStore as VectorStoreFromLC
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [langchain] not found.\nRun 'pip install \"beeai-framework[langchain]\"' to install."
    ) from e

from beeai_framework.adapters.langchain.mappers.documents import document_to_lc_document, lc_document_to_document
from beeai_framework.backend.embedding import EmbeddingModel
from beeai_framework.backend.vector_store import VectorStore
from beeai_framework.logger import Logger


class LangChainVectorStore(VectorStore):
    integration_name: str = "langchain"

    def __init__(self, langchain_vector_store: VectorStoreFromLC) -> None:
        self.vector_store: VectorStoreFromLC = langchain_vector_store

    async def add_documents(self, documents: Document) -> list[str]:
        if self.vector_store is None:
            raise ValueError("Vector store must be set before adding documents")
        lc_documents = [document_to_lc_document(document) for document in documents]
        return await self.vector_store.aadd_documents(lc_documents)

    async def search(self, query: str, k: int = 4, **kwargs: Any) -> list[DocumentWithScore]:
        if self.vector_store is None:
            raise ValueError("Vector store must be set before searching for documents")
        lc_documents_with_scores: list[
            tuple[LCDocument, float]
        ] = await self.vector_store.asimilarity_search_with_score(query=query, k=k, **kwargs)
        documents_with_scores = [
            DocumentWithScore(document=lc_document_to_document(lc_document), score=score)
            for lc_document, score in lc_documents_with_scores
        ]
        return documents_with_scores

    @classmethod
    def _class_from_name(cls, class_name: str, embedding_model: EmbeddingModel, **kwargs: Any) -> LangChainVectorStore:
        """
        Dynamically imports and instantiates `class_name` from all vector store paths on LangChain
        """
        lc_vector_store = None
        # Convert BeeAI embedding to LangChain embedding
        lc_embedding = LangChainBeeAIEmbeddingModel(embedding_model)
        # InMemoryVectorStore is loaded the core LC vectorstores
        if class_name == "InMemoryVectorStore":
            from langchain_core.vectorstores import InMemoryVectorStore

            lc_vector_store = InMemoryVectorStore(embedding=lc_embedding)

        # Then try the LangChain pattern for integrations with langchain-*
        # This and the LangChain vector store approaches should be integrated in the future, see LangChain discussion
        # https://github.com/langchain-ai/langchain/discussions/31807
        if lc_vector_store is None:
            try:
                module_name = f"langchain_{class_name.lower()}"
                module = importlib.import_module(module_name)
                cls_obj = getattr(module, class_name)
                lc_vector_store = cls_obj(embedding_function=lc_embedding, **kwargs)
            except (ImportError, AttributeError):
                Logger.info(
                    f"Failed to import LangChain vector store {class_name} from the external integrations, \
                            trying standard LangChain integration"
                )

        # Final resort it to import from LangChain
        if lc_vector_store is None:
            try:
                module_name = "langchain.vectorstores"
                module = importlib.import_module(module_name)
                cls_obj = getattr(module, class_name)
                lc_vector_store = cls_obj(embedding_function=lc_embedding, **kwargs)
            except (ImportError, AttributeError):
                Logger.error(f"Failed to import class {class_name}")
                return None

        return cls(langchain_vector_store=lc_vector_store)
