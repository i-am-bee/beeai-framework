# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import os

from typing_extensions import Unpack

from beeai_framework.adapters.litellm import utils
from beeai_framework.adapters.litellm.embedding import LiteLLMEmbeddingModel
from beeai_framework.backend.constants import ProviderName
from beeai_framework.backend.embedding import EmbeddingModelKwargs


class GeminiEmbeddingModel(LiteLLMEmbeddingModel):
    @property
    def provider_id(self) -> ProviderName:
        return "gemini"

    def __init__(
        self,
        model_id: str | None = None,
        *,
        api_key: str | None = None,
        **kwargs: Unpack[EmbeddingModelKwargs],
    ) -> None:
        super().__init__(
            model_id if model_id else os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"),
            provider_id="gemini",
            **kwargs,
        )
        self._assert_setting_value("api_key", api_key, envs=["GEMINI_API_KEY"])
        self._settings["extra_headers"] = utils.parse_extra_headers(
            self._settings.get("extra_headers"), os.getenv("GEMINI_API_HEADERS")
        )
