# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import os

from typing_extensions import Unpack

from beeai_framework.adapters.litellm import utils
from beeai_framework.adapters.litellm.chat import LiteLLMChatModel
from beeai_framework.backend.chat import ChatModelKwargs
from beeai_framework.backend.constants import ProviderName
from beeai_framework.logger import Logger

logger = Logger(__name__)


class AmazonBedrockChatModel(LiteLLMChatModel):
    @property
    def provider_id(self) -> ProviderName:
        return "amazon_bedrock"

    def __init__(
        self,
        model_id: str | None = None,
        *,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        region: str | None = None,
        **kwargs: Unpack[ChatModelKwargs],
    ) -> None:
        super().__init__(
            (model_id if model_id else os.getenv("AWS_CHAT_MODEL", "llama-3.1-8b-instant")),
            provider_id="bedrock",
            **kwargs,
        )

        self._assert_setting_value(
            "aws_access_key_id", access_key_id, display_name="access_key_id", envs=["AWS_ACCESS_KEY_ID"]
        )
        self._assert_setting_value(
            "aws_secret_access_key",
            secret_access_key,
            display_name="secret_access_key",
            envs=["AWS_SECRET_ACCESS_KEY"],
        )
        self._assert_setting_value(
            "aws_region_name", region, envs=["AWS_REGION", "AWS_REGION_NAME"], display_name="region"
        )

        self._settings["extra_headers"] = utils.parse_extra_headers(
            self._settings.get("extra_headers"), os.getenv("AWS_API_HEADERS")
        )
