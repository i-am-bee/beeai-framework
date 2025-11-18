# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Unpack

from beeai_framework.adapters.litellm import LiteLLMChatModel, utils
from beeai_framework.backend.chat import ChatModelKwargs
from beeai_framework.backend.constants import ProviderName
from beeai_framework.logger import Logger

logger = Logger(__name__)


class DeepseekChatModel(LiteLLMChatModel):
    @property
    def provider_id(self) -> ProviderName:
        return "deepseek"

    def __init__(
        self,
        model_id: str | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Unpack[ChatModelKwargs],
    ) -> None:
        super().__init__(
            model_id if model_id else os.getenv("DEEPSEEK_CHAT_MODEL", "deepseek-chat"),
            provider_id="deepseek",
            **kwargs,
        )

        self._assert_setting_value("api_key", api_key, envs=["DEEPSEEK_API_KEY"])
        self._assert_setting_value(
            "base_url", base_url, envs=["DEEPSEEK_API_BASE"], aliases=["api_base"], allow_empty=True
        )
        self._settings["extra_headers"] = utils.parse_extra_headers(
            self._settings.get("extra_headers"), os.getenv("DEEPSEEK_API_HEADERS")
        )
