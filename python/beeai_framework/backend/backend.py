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
