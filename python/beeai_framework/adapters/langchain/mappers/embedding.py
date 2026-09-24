# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from beeai_framework.backend.embedding import EmbeddingModel
from beeai_framework.backend.types import EmbeddingModelOutput
from beeai_framework.logger import Logger
from beeai_framework.utils.asynchronous import run_sync

try:
    from langchain_core.embeddings import Embeddings as LCEmbeddingModel
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [langchain] not found.\nRun 'pip install \"beeai-framework[langchain]\"' to install."
    ) from e

logger = Logger(__name__)


class LangChainBeeAIEmbeddingModel(LCEmbeddingModel):
    def __init__(self, embedding: EmbeddingModel, batch_size: int = 1000) -> None:
        self._embedding_model = embedding
        self._batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embedding_res: EmbeddingModelOutput = run_sync(self._embedding_model.create(values=texts))
        return embedding_res.embeddings

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        results = []
        for iteration_start_index in range(0, len(texts), self._batch_size):
            texts_to_embed = texts[iteration_start_index : iteration_start_index + self._batch_size]
            embedding_res: EmbeddingModelOutput = await self._embedding_model.create(values=texts_to_embed)
            results.extend(embedding_res.embeddings)
        return results

    def embed_query(self, text: str) -> list[float]:
        embedding_res: EmbeddingModelOutput = run_sync(self._embedding_model.create(values=[text]))
        return embedding_res.embeddings[0]

    async def aembed_query(self, text: str) -> list[float]:
        embedding_res: EmbeddingModelOutput = await self._embedding_model.create(values=[text])
        return embedding_res.embeddings[0]
