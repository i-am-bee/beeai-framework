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
from typing import Any

from beeai_framework.adapters.litellm.embedding import LiteLLMEmbeddingModel
from beeai_framework.backend.constants import ProviderName


class OllamaEmbeddingModel(LiteLLMEmbeddingModel):
    @property
    def provider_id(self) -> ProviderName:
        return "ollama"

    def __init__(
        self,
        model_id: str | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_id if model_id else os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
            provider_id="openai",
            **kwargs,
        )

        self._assert_setting_value("api_key", api_key, envs=["OLLAMA_API_KEY"], fallback="ollama")
        self._assert_setting_value(
            "base_url", base_url, envs=["OLLAMA_API_BASE"], fallback="http://localhost:11434", aliases=["api_base"]
        )
        if not self._settings["base_url"].endswith("/v1"):
            self._settings["base_url"] += "/v1"
