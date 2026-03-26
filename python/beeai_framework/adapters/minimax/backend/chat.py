# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import os
from typing_extensions import Unpack

from beeai_framework.adapters.litellm import LiteLLMChatModel, utils
from beeai_framework.backend.chat import ChatModelKwargs
from beeai_framework.backend.constants import ProviderName
from beeai_framework.logger import Logger

logger = Logger(__name__)

MINIMAX_API_BASE = "https://api.minimax.io/v1"


class MiniMaxChatModel(LiteLLMChatModel):
    """
    A chat model implementation for the MiniMax provider, leveraging LiteLLM.

    MiniMax provides an OpenAI-compatible API. This adapter routes requests
    through LiteLLM's OpenAI provider with the MiniMax base URL.

    Available models include MiniMax-M2.7, MiniMax-M2.7-highspeed,
    MiniMax-M2.5, and MiniMax-M2.5-highspeed.
    """

    @property
    def provider_id(self) -> ProviderName:
        """The provider ID for MiniMax."""
        return "minimax"

    def __init__(
        self,
        model_id: str | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Unpack[ChatModelKwargs],
    ) -> None:
        """
        Initializes the MinimaxChatModel.

        Args:
            model_id: The ID of the MiniMax model to use. If not provided,
                it falls back to the MINIMAX_CHAT_MODEL environment variable,
                and then defaults to 'MiniMax-M2.7'.
            api_key: The MiniMax API key. Falls back to MINIMAX_API_KEY env var.
            base_url: The MiniMax API base URL. Falls back to MINIMAX_API_BASE
                env var, then defaults to 'https://api.minimax.io/v1'.
            **kwargs: Additional settings to configure the provider.
        """
        super().__init__(
            model_id if model_id else os.getenv("MINIMAX_CHAT_MODEL", "MiniMax-M2.7"),
            provider_id="openai",
            **kwargs,
        )

        self._assert_setting_value("api_key", api_key, envs=["MINIMAX_API_KEY"])
        self._assert_setting_value(
            "base_url",
            base_url,
            envs=["MINIMAX_API_BASE"],
            aliases=["api_base"],
            allow_empty=True,
            fallback=MINIMAX_API_BASE,
        )
        self._settings["extra_headers"] = utils.parse_extra_headers(
            self._settings.get("extra_headers"), os.getenv("MINIMAX_API_HEADERS")
        )
