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


class OpenAIChatModel(LiteLLMChatModel):
    """
    A chat model implementation for the OpenAI provider, leveraging LiteLLM.
    """

    @property
    def provider_id(self) -> ProviderName:
        """The provider ID for OpenAI."""
        return "openai"

    def __init__(
        self,
        model_id: str | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        text_completion: bool | None = False,
        **kwargs: Unpack[ChatModelKwargs],
    ) -> None:
        """
        Initializes the OpenAIChatModel.

        Args:
            model_id: The ID of the OpenAI model to use. If not provided,
                it falls back to the OPENAI_CHAT_MODEL environment variable,
                and then defaults to 'gpt-4o'.
            **kwargs: A dictionary of settings to configure the provider.
        """
        super().__init__(
            model_id if model_id else os.getenv("OPENAI_CHAT_MODEL", "gpt-4o"),
            provider_id="text-completion-openai" if text_completion else "openai",
            **kwargs,
        )
        self._assert_setting_value("api_key", api_key, envs=["OPENAI_API_KEY"])
        self._assert_setting_value(
            "base_url", base_url, envs=["OPENAI_API_BASE"], aliases=["api_base"], allow_empty=True
        )
        self._settings["extra_headers"] = utils.parse_extra_headers(
            self._settings.get("extra_headers"), os.getenv("OPENAI_API_HEADERS")
        )
