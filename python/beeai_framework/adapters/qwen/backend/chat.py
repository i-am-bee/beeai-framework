# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Unpack

from beeai_framework.adapters.litellm import LiteLLMChatModel, utils
from beeai_framework.backend.chat import ChatModelKwargs
from beeai_framework.backend.constants import ProviderName
from beeai_framework.logger import Logger

logger = Logger(__name__)


class QwenChatModel(LiteLLMChatModel):
    @property
    def provider_id(self) -> ProviderName:
        return "qwen"

    def __init__(
        self,
        model_id: str | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Unpack[ChatModelKwargs],
    ) -> None:
        model_id = model_id if model_id else os.getenv("DASHSCOPE_CHAT_MODEL", "qwen-plus")
        # pyrefly: ignore [missing-attribute]
        model_id = model_id.removeprefix("dashscope/")

        super().__init__(
            f"dashscope/{model_id}",
            provider_id="openai",  # LiteLLM expects openai for all Qwen models
            **kwargs,
        )

        self._assert_setting_value("api_key", api_key, envs=["DASHSCOPE_API_KEY"])
        self._assert_setting_value(
            "base_url", base_url, envs=["DASHSCOPE_API_BASE"], aliases=["api_base"], allow_empty=True
        )
        self._settings["extra_headers"] = utils.parse_extra_headers(
            self._settings.get("extra_headers"), os.getenv("DASHSCOPE_API_HEADERS")
        )
