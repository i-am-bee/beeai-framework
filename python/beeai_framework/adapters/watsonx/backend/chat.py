# Copyright 2025 IBM Corp.
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
from typing import Any, ClassVar

from beeai_framework.adapters.litellm.chat import LiteLLMChatModel
from beeai_framework.backend.constants import ProviderName
from beeai_framework.logger import Logger

logger = Logger(__name__)


class WatsonxChatModel(LiteLLMChatModel):
    tool_choice_support: ClassVar[set[str]] = {"none", "single", "auto"}

    @property
    def provider_id(self) -> ProviderName:
        return "watsonx"

    # Differs between typescript & litellm so directly set relevant property here if env specified -> litellm:
    # Litellm docs have WATSONX_APIKEY. This does not work. It's WATSONX_API_KEY . See     https://github.com/BerriAI/litellm/issues/7595
    # WATSONX_SPACE_ID [WATSONX_DEPLOYMENT_SPACE_ID] -> space_id

    # LiteLLM uses 'url', from WATSONX_URL
    # If set we use that (as more specific), otherwise compose from WATSONX_REGION & a constant base.

    # Extra for LiteLLM - no code here - passthrough
    # WATSONX_TOKEN (not in ts)
    # WATSONX_ZENAPIKEY (not in ts)
    # WATSONX_URL

    # https://docs.litellm.ai/docs/providers/watsonx

    def __init__(self, model_id: str | None = None, settings: dict[str, Any] | None = None) -> None:
        _settings = settings.copy() if settings is not None else {}

        # Set space_id only if not already in settings
        if "space_id" not in _settings or not _settings["space_id"]:
            watsonx_space_id = os.getenv("WATSONX_SPACE_ID")
            if watsonx_space_id:
                _settings["space_id"] = watsonx_space_id

        # Set URL based on priority: existing setting > WATSONX_URL env > region-based > error
        if "url" not in _settings or not _settings["url"]:
            watsonx_url = os.getenv("WATSONX_URL")
            if watsonx_url:
                _settings["url"] = watsonx_url
            else:
                watsonx_region = os.getenv("WATSONX_REGION")
                if watsonx_region:
                    _settings["url"] = f"https://{watsonx_region}.ml.cloud.ibm.com"
                else:
                    raise ValueError(
                        "Watsonx URL not set. Please provide a 'url' in settings, "
                        "set the WATSONX_URL environment variable, "
                        "or set the WATSONX_REGION environment variable."
                    )

        super().__init__(
            model_id if model_id else os.getenv("WATSONX_CHAT_MODEL", "ibm/granite-3-8b-instruct"),
            provider_id="watsonx",
            settings=_settings,
        )
