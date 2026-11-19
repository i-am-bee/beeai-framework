# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any

import litellm
from dotenv import load_dotenv
from litellm.llms.anthropic.chat.transformation import AnthropicConfig
from typing_extensions import Unpack

from beeai_framework.adapters.litellm import utils
from beeai_framework.adapters.litellm.chat import LiteLLMChatModel
from beeai_framework.backend.chat import ChatModelKwargs
from beeai_framework.backend.constants import ProviderName
from beeai_framework.backend.types import ChatModelInput, ChatModelParameters
from beeai_framework.logger import Logger

logger = Logger(__name__)
load_dotenv()

# Claude models that answer a forced `tool_choice` (`any` / `tool`) with an HTTP 400.
# Matched as a prefix so dated snapshots are covered while `claude-opus-5` / `claude-sonnet-5` are not.
_NO_FORCED_TOOL_CHOICE_PREFIXES = (
    "claude-opus-5-5",
    "claude-sonnet-5-5",
    "claude-fable-5-1",
    "claude-mythos-5-1",
)


def _register_output_config_support(model_id: str) -> None:
    """Tell LiteLLM the model accepts `output_config`, which it otherwise drops for models it does not know."""
    name = model_id.removeprefix("anthropic/")
    if name in litellm.model_cost or f"anthropic/{name}" in litellm.model_cost:
        return  # LiteLLM already knows the model; do not overwrite its entry
    litellm.register_model({name: {"litellm_provider": "anthropic", "mode": "chat", "supports_output_config": True}})


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

        if self._rejects_forced_tool_choice and kwargs.get("tool_choice_support") is None:
            self._tool_choice_support.discard("required")
            self._tool_choice_support.discard("single")

        if self._rejects_forced_tool_choice:
            _register_output_config_support(self.model_id)

    @property
    def _rejects_forced_tool_choice(self) -> bool:
        return self.model_id.removeprefix("anthropic/").startswith(_NO_FORCED_TOOL_CHOICE_PREFIXES)

    @classmethod
    def get_default_parameters(cls) -> ChatModelParameters:
        # Leaves `temperature` unset, so `_transform_input` can tell the implicit 0 from a chosen one
        return ChatModelParameters()

    def _transform_input(self, input: ChatModelInput) -> dict[str, Any]:
        params = super()._transform_input(input)

        self._drop_implicit_temperature(input, params)

        # Without forced tool choice `ChatModel` asks for the tool call through the response format.
        # Send it as Anthropic's structured outputs (`output_config.format`).
        if self._rejects_forced_tool_choice and input.response_format:
            output_format = AnthropicConfig().map_response_format_to_anthropic_output_format(
                params.get("response_format")
            )
            if output_format:
                params.pop("response_format", None)
                params["output_config"] = {**(params.get("output_config") or {}), "format": dict(output_format)}

        return params
