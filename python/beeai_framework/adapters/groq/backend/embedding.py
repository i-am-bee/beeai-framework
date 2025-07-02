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


import os

from typing_extensions import Unpack

from beeai_framework.adapters.litellm.embedding import LiteLLMEmbeddingModel
from beeai_framework.backend.constants import ProviderName
from beeai_framework.backend.embedding import EmbeddingModelKwargs


class GroqEmbeddingModel(LiteLLMEmbeddingModel):
    @property
    def provider_id(self) -> ProviderName:
        return "groq"

    def __init__(
        self,
        model_id: str | None = None,
        *,
        api_key: str | None = None,
        **kwargs: Unpack[EmbeddingModelKwargs],
    ) -> None:
        super().__init__(
            model_id if model_id else os.getenv("GROQ_EMBEDDING_MODEL", "llama-3.1-8b-instant"),
            provider_id="groq",
            **kwargs,
        )

        self._assert_setting_value("api_key", api_key, envs=["GROQ_API_KEY"])
