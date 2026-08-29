# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import os

from typing_extensions import Unpack

from beeai_framework.adapters.litellm import LiteLLMChatModel, utils
from beeai_framework.backend.chat import ChatModelKwargs
from beeai_framework.backend.constants import ProviderName
from beeai_framework.logger import Logger

logger = Logger(__name__)

ORCAROUTER_API_BASE = "https://api.orcarouter.ai/v1"


class OrcaRouterChatModel(LiteLLMChatModel):
    """
    A chat model implementation for the OrcaRouter provider, leveraging LiteLLM.

    OrcaRouter exposes an OpenAI-compatible API. This adapter routes requests
    through LiteLLM's OpenAI provider with the OrcaRouter base URL.

    Models are addressed as ``orcarouter/auto`` (the smart routing router) or
    any model from the public catalog, e.g. ``deepseek/deepseek-v4-flash``.
    """

    @property
    def provider_id(self) -> ProviderName:
        return "orcarouter"

    def __init__(
        self,
        model_id: str | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Unpack[ChatModelKwargs],
    ) -> None:
        super().__init__(
            model_id if model_id else os.getenv("ORCAROUTER_CHAT_MODEL", "orcarouter/auto"),
            provider_id="openai",
            **kwargs,
        )

        self._assert_setting_value("api_key", api_key, envs=["ORCAROUTER_API_KEY"])
        self._assert_setting_value(
            "base_url",
            base_url,
            envs=["ORCAROUTER_API_BASE"],
            aliases=["api_base"],
            allow_empty=True,
            fallback=ORCAROUTER_API_BASE,
        )
        self._settings["extra_headers"] = utils.parse_extra_headers(
            self._settings.get("extra_headers"), os.getenv("ORCAROUTER_API_HEADERS")
        )
