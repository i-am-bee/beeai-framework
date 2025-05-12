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
        **kwargs: Any,
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
