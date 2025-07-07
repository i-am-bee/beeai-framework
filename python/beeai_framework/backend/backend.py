# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.constants import ProviderName
from beeai_framework.backend.embedding import EmbeddingModel


class Backend:
    def __init__(self, *, chat: ChatModel, embedding: EmbeddingModel) -> None:
        self.chat = chat
        self.embedding = embedding

    @staticmethod
    def from_name(*, chat: str | ProviderName, embedding: str | ProviderName) -> "Backend":
        return Backend(chat=ChatModel.from_name(chat), embedding=EmbeddingModel.from_name(embedding))

    @staticmethod
    def from_provider(name: str | ProviderName) -> "Backend":
        return Backend.from_name(chat=name, embedding=name)

    async def clone(self) -> "Backend":
        return Backend(chat=await self.chat.clone(), embedding=self.embedding.clone())
