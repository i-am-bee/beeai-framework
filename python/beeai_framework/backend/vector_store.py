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

from abc import ABC, abstractmethod
from typing import Any

from beeai_framework.backend.embedding import EmbeddingModel
from beeai_framework.backend.types import Document, DocumentWithScore


class VectorStore(ABC):
    def __init__(self, embedding_model: EmbeddingModel = None) -> None:
        self.embedding_model = embedding_model

    @abstractmethod
    async def aadd_documents(self, documents: Document) -> list[str]:
        raise NotImplementedError("Implement me")

    @abstractmethod
    async def asearch(self, query: str, search_type: str, k: int = 4, **kwargs: Any) -> list[DocumentWithScore]:
        raise NotImplementedError("Implement me")
