# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import os

from typing_extensions import Unpack

from beeai_framework.adapters.litellm.embedding import LiteLLMEmbeddingModel
from beeai_framework.backend.constants import ProviderName
from beeai_framework.backend.embedding import EmbeddingModelKwargs


class WatsonxEmbeddingModel(LiteLLMEmbeddingModel):
    @property
    def provider_id(self) -> ProviderName:
        return "watsonx"

    def __init__(
        self,
        model_id: str | None = None,
        *,
        api_key: str | None = None,
        project_id: str | None = None,
        space_id: str | None = None,
        region: str | None = None,
        base_url: str | None = None,
        **kwargs: Unpack[EmbeddingModelKwargs],
    ) -> None:
        super().__init__(
            model_id if model_id else os.getenv("WATSONX_EMBEDDING_MODEL", "ibm/granite-embedding-107m-multilingual"),
            provider_id="watsonx",
            **kwargs,
        )

        self._assert_setting_value(
            "space_id", space_id, envs=["WATSONX_SPACE_ID", "WATSONX_DEPLOYMENT_SPACE_ID"], allow_empty=True
        )
        if not self._settings.get("space_id"):
            self._assert_setting_value("project_id", project_id, envs=["WATSONX_PROJECT_ID"])

        self._assert_setting_value("region", region, envs=["WATSONX_REGION"], fallback="us-south")
        self._assert_setting_value(
            "base_url",
            base_url,
            aliases=["api_base"],
            envs=["WATSONX_URL"],
            fallback=f"https://{self._settings['region']}.ml.cloud.ibm.com",
        )
        self._assert_setting_value(
            "api_key",
            api_key,
            envs=["WATSONX_API_KEY", "WATSONX_APIKEY", "WATSONX_ZENAPIKEY"],
            allow_empty=True,
        )
