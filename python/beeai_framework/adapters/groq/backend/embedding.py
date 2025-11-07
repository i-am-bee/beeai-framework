# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

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
