# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any

from dotenv import load_dotenv
from typing_extensions import Unpack

from beeai_framework.adapters.litellm import utils
from beeai_framework.adapters.litellm.chat import LiteLLMChatModel
from beeai_framework.backend.chat import ChatModelKwargs
from beeai_framework.backend.constants import ProviderName
from beeai_framework.backend.types import ChatModelInput, ChatModelParameters
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
            (model_id if model_id else os.getenv("ANTHROPIC_CHAT_MODEL", "claude-haiku-4-5")),
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

    @classmethod
    def get_default_parameters(cls) -> ChatModelParameters:
        # Leaves `temperature` unset, so `_transform_input` can tell the implicit 0 from a chosen one
        return ChatModelParameters()

    def _transform_input(self, input: ChatModelInput) -> dict[str, Any]:
        params = super()._transform_input(input)

        # `temperature` defaults to 0 for every provider, but Anthropic rejects it next to
        # `top_p` and requires 1 when thinking is on. Only a value nobody asked for is dropped.
        temperature_is_explicit = (
            "temperature" in input.model_fields_set
            or "temperature" in self.parameters.model_fields_set
            or "temperature" in self._settings
        )
        if not temperature_is_explicit and any(key in params for key in ("top_p", "reasoning_effort", "thinking")):
            params.pop("temperature", None)

        return params
