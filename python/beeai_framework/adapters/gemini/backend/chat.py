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


class GeminiChatModel(LiteLLMChatModel):
    @property
    def provider_id(self) -> ProviderName:
        return "gemini"

    def __init__(
        self,
        model_id: str | None = None,
        *,
        api_key: str | None = None,
        **kwargs: Unpack[ChatModelKwargs],
    ) -> None:
        super().__init__(
            model_id if model_id else os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash"),
            provider_id="gemini",
            **kwargs,
        )
        self._assert_setting_value("api_key", api_key, envs=["GEMINI_API_KEY"])
        self._settings["extra_headers"] = utils.parse_extra_headers(
            self._settings.get("extra_headers"), os.getenv("GEMINI_API_HEADERS")
        )
