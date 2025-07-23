# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod

from beeai_framework.backend.types import DocumentWithScore


class DocumentProcessor(ABC):
    @abstractmethod
    async def postprocess_documents(
        self, documents: list[DocumentWithScore], *, query: str | None = None
    ) -> list[DocumentWithScore]:
        raise NotImplementedError()
