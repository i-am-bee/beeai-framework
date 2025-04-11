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

# Import necessary types from base classes and pydantic
from pydantic import BaseModel

from beeai_framework.adapters.litellm.chat import LiteLLMChatModel
from beeai_framework.backend.constants import ProviderName

# Assuming ChatModelInput is needed for _transform_input override later
from beeai_framework.logger import Logger

logger = Logger(__name__)


class WatsonxChatModel(LiteLLMChatModel):
    tool_choice_support: ClassVar[set[str]] = {"none", "single", "auto"}
    # TODO: WatsonX seems to require 'required' for tool_choice when forcing a tool call

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

        # Set project_id (LiteLLM uses project_id for WatsonX)
        if "project_id" not in _settings or not _settings["project_id"]:
            watsonx_project_id = os.getenv("WATSONX_PROJECT_ID")
            if watsonx_project_id:
                _settings["project_id"] = watsonx_project_id

        # Set space_id only if not already in settings - also used in watsonx
        if "space_id" not in _settings or not _settings["space_id"]:
            watsonx_space_id = os.getenv("WATSONX_SPACE_ID")
            if watsonx_space_id:
                _settings["space_id"] = watsonx_space_id

        # Set URL (api_base for LiteLLM)
        if "api_base" not in _settings or not _settings["api_base"]:
            watsonx_url = os.getenv("WATSONX_URL")
            if watsonx_url:
                _settings["api_base"] = watsonx_url
            else:
                watsonx_region = os.getenv("WATSONX_REGION")
                if watsonx_region:
                    _settings["api_base"] = f"https://{watsonx_region}.ml.cloud.ibm.com"
                else:
                    raise ValueError(
                        "Watsonx api_base not set. Please provide a 'api_base' in settings, "
                        "set the WATSONX_URL environment variable, "
                        "or set the WATSONX_REGION environment variable."
                    )
        # TODO: Consider validating other sufficient parms are provided

        super().__init__(
            model_id=model_id if model_id else os.getenv("WATSONX_CHAT_MODEL", "ibm/granite-3-8b-instruct"),
            provider_id="watsonx",
            settings=_settings,
        )

    def _format_response_model(
        self, model: type[BaseModel] | dict[str, Any] | None
    ) -> type[BaseModel] | dict[str, Any]:
        """
        Based on the error "Missing json field response_format.type", watsonx
        return {'type': 'json_object'} when a format is requested.
        """
        if model is None:
            # If no response format schema was provided in the input, return empty dict.
            return {}

        # Regardless of the input schema details (BaseModel type or dict),
        # return the specific dictionary that WatsonX expects to enable JSON mode.
        logger.debug("Formatting response_format for WatsonX: {'type': 'json_object'}")
        return {"type": "json_object"}
