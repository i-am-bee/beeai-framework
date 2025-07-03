# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import os
from abc import ABC
from itertools import chain
from typing import Any

import litellm
from litellm import aembedding
from litellm.litellm_core_utils.get_supported_openai_params import get_supported_openai_params
from litellm.types.utils import EmbeddingResponse
from typing_extensions import Unpack

from beeai_framework.adapters.litellm.utils import litellm_debug
from beeai_framework.backend import EmbeddingModel
from beeai_framework.backend.embedding import EmbeddingModelKwargs
from beeai_framework.backend.types import EmbeddingModelInput, EmbeddingModelOutput, EmbeddingModelUsage
from beeai_framework.context import RunContext
from beeai_framework.logger import Logger

logger = Logger(__name__)


class LiteLLMEmbeddingModelOutput(EmbeddingModelOutput):
    response: Any


class LiteLLMEmbeddingModel(EmbeddingModel, ABC):
    def __init__(
        self,
        model_id: str,
        *,
        provider_id: str,
        **kwargs: Unpack[EmbeddingModelKwargs],
    ) -> None:
        super().__init__(**kwargs)
        self._model_id = model_id
        self._litellm_provider_id = provider_id
        self.supported_params = get_supported_openai_params(model=self.model_id, custom_llm_provider=provider_id) or []
        # drop any unsupported parameters that were passed in
        litellm.drop_params = True
        # disable LiteLLM caching in favor of our own
        litellm.disable_cache()  # type: ignore [attr-defined]

    @property
    def model_id(self) -> str:
        return self._model_id

    async def _create(
        self,
        input: EmbeddingModelInput,
        run: RunContext,
    ) -> EmbeddingModelOutput:
        litellm_input = self._transform_input(input)
        response = await aembedding(**litellm_input)
        response_output = self._transform_output(response, input)
        logger.debug(f"Inference response output:\n{response_output}")
        return response_output

    def _transform_input(self, model_input: EmbeddingModelInput) -> dict[str, Any]:
        return {
            "model": f"{self._litellm_provider_id}/{self._model_id}",
            "input": model_input.values,
            **self._settings,
        }

    def _transform_output(
        self, response: EmbeddingResponse, model_input: EmbeddingModelInput
    ) -> LiteLLMEmbeddingModelOutput:
        embeddings: list[list[float]] = []

        for result in response.data:
            embeddings.append([float(value) for value in result.get("embedding")])

        return LiteLLMEmbeddingModelOutput(
            values=model_input.values,
            embeddings=embeddings,
            usage=EmbeddingModelUsage(**response.usage.model_dump()) if response.usage else None,
            response=response,
        )

    def _assert_setting_value(
        self,
        name: str,
        value: Any | None = None,
        *,
        display_name: str | None = None,
        aliases: list[str] | None = None,
        envs: list[str],
        fallback: str | None = None,
        allow_empty: bool = False,
    ) -> None:
        aliases = aliases or []
        assert aliases is not None

        value = value or self._settings.get(name)
        if not value:
            value = next(
                chain(
                    (self._settings[alias] for alias in aliases if self._settings.get(alias)),
                    (os.environ[env] for env in envs if os.environ.get(env)),
                ),
                fallback,
            )

        for alias in aliases:
            self._settings[alias] = None

        if not value and not allow_empty:
            raise ValueError(
                f"Setting {display_name or name} is required for {type(self).__name__}. "
                f"Either pass the {display_name or name} explicitly or set one of the "
                f"following environment variables: {', '.join(envs)}."
            )

        self._settings[name] = value or None


litellm_debug(False)
