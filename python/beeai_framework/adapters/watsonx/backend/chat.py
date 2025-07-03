# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import os
from typing import ClassVar

from typing_extensions import Unpack

from beeai_framework.adapters.litellm.chat import LiteLLMChatModel
from beeai_framework.backend import ChatModelParameters
from beeai_framework.backend.chat import ChatModelKwargs
from beeai_framework.backend.constants import ProviderName
from beeai_framework.logger import Logger

logger = Logger(__name__)


class WatsonxChatModel(LiteLLMChatModel):
    tool_choice_support: ClassVar[set[str]] = {"none", "single", "auto"}

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
        **kwargs: Unpack[ChatModelKwargs],
    ) -> None:
        super().__init__(
            model_id if model_id else os.getenv("WATSONX_CHAT_MODEL", "ibm/granite-3-8b-instruct"),
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

    @classmethod
    def get_default_parameters(cls) -> ChatModelParameters:
        # Source: https://cloud.ibm.com/apidocs/watsonx-ai#text-chat
        return ChatModelParameters(
            temperature=0,  # initially 1
            max_tokens=4096,  # initially 1024
            n=1,
            frequency_penalty=0,
            top_p=1,
        )
