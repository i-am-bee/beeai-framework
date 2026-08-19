# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing_extensions import Unpack

from beeai_framework.adapters.litellm.embedding import LiteLLMEmbeddingModel
from beeai_framework.backend.constants import ProviderName
from beeai_framework.backend.embedding import EmbeddingModelKwargs


class GroqEmbeddingModel(LiteLLMEmbeddingModel):
    """Groq does not expose a text embeddings API.

    This class exists only to surface a clear, actionable error instead of a cryptic
    provider failure when embeddings are requested for the ``groq`` provider (for example
    via ``EmbeddingModel.from_name("groq")``). The previous default ``model_id`` pointed at
    ``llama-3.1-8b-instant``, which is a Groq *chat* model and cannot serve embeddings.
    """

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
        raise NotImplementedError(
            "Groq does not provide an embeddings API, so embeddings are not supported for the 'groq' provider. "
            "Use a provider that offers embeddings instead (e.g. OpenAI, Ollama, watsonx, Gemini, or MistralAI)."
        )
