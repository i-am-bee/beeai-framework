# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import os

from dotenv import load_dotenv
from typing_extensions import Unpack

from beeai_framework.adapters.litellm import utils
from beeai_framework.adapters.litellm.chat import LiteLLMChatModel
from beeai_framework.backend.chat import ChatModelKwargs
from beeai_framework.backend.constants import ProviderName
from beeai_framework.logger import Logger

logger = Logger(__name__)
load_dotenv()


class AnthropicChatModel(LiteLLMChatModel):
    @property
    def provider_id(self) -> ProviderName:
        return "anthropic"

    def __init__(
        self,
        model_id: str | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Unpack[ChatModelKwargs],
    ) -> None:
        super().__init__(
            (model_id if model_id else os.getenv("ANTHROPIC_CHAT_MODEL", "claude-3-haiku-20240307")),
            provider_id="anthropic",
            **kwargs,
        )

        self._assert_setting_value("api_key", api_key, envs=["ANTHROPIC_API_KEY"])
        self._assert_setting_value(
            "base_url", base_url, envs=["ANTHROPIC_API_BASE"], aliases=["api_base"], allow_empty=True
        )
        self._settings["extra_headers"] = utils.parse_extra_headers(
            self._settings.get("extra_headers"), os.getenv("ANTHROPIC_API_HEADERS")
        )
