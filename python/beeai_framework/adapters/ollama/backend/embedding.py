# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import os

from typing_extensions import Unpack

from beeai_framework.adapters.litellm.embedding import LiteLLMEmbeddingModel
from beeai_framework.backend.constants import ProviderName
from beeai_framework.backend.embedding import EmbeddingModelKwargs


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
        **kwargs: Unpack[EmbeddingModelKwargs],
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
