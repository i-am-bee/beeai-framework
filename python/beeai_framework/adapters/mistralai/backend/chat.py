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


class MistralAIChatModel(LiteLLMChatModel):
    """
    A chat model implementation for the MistralAI provider, leveraging LiteLLM.
    """

    @property
    def provider_id(self) -> ProviderName:
        """The provider ID for MistralAI."""
        return "mistralai"

    def __init__(
        self,
        model_id: str | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Unpack[ChatModelKwargs],
    ) -> None:
        """
        Initializes the MistralAIChatModel.
        """
        super().__init__(
            model_id if model_id else os.getenv("MISTRALAI_CHAT_MODEL", "mistral-tiny"),
            provider_id="mistral",
            **kwargs,
        )
        self._assert_setting_value("api_key", api_key, envs=["MISTRALAI_API_KEY"])
        self._assert_setting_value(
            "base_url", base_url, envs=["MISTRALAI_API_BASE"], aliases=["api_base"], allow_empty=True
        )
        self._settings["extra_headers"] = utils.parse_extra_headers(
            self._settings.get("extra_headers"), os.getenv("MISTRALAI_API_HEADERS")
        )
