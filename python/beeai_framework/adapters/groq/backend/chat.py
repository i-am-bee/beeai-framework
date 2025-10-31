# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import copy
import os
from typing import Any, Self

from litellm.exceptions import BadRequestError
from pydantic import BaseModel
from typing_extensions import Unpack, override

from beeai_framework.adapters.litellm.chat import LiteLLMChatModel
from beeai_framework.backend import AssistantMessage, ChatModelOutput
from beeai_framework.backend.chat import ChatModelKwargs
from beeai_framework.backend.constants import ProviderName
from beeai_framework.backend.types import ChatModelInput
from beeai_framework.backend.utils import inline_schema_refs, parse_broken_json
from beeai_framework.context import RunContext
from beeai_framework.logger import Logger
from beeai_framework.utils.models import is_pydantic_model
from beeai_framework.utils.schema import SimplifyJsonSchemaConfig, simplify_json_schema
from beeai_framework.utils.strings import find_first_pair

logger = Logger(__name__)


class GroqChatModel(LiteLLMChatModel):
    @property
    def provider_id(self) -> ProviderName:
        return "groq"

    def __init__(
        self,
        model_id: str | None = None,
        api_key: str | None = None,
        *,
        fallback_failed_generation: bool = False,
        **kwargs: Unpack[ChatModelKwargs],
    ) -> None:
        super().__init__(
            model_id if model_id else os.getenv("GROQ_CHAT_MODEL", "llama-3.1-8b-instant"),
            provider_id="groq",
            **kwargs,
        )
        self._assert_setting_value("api_key", api_key, envs=["GROQ_API_KEY"])
        self._fallback_failed_generation = fallback_failed_generation

    @override
    async def _create(
        self,
        input: ChatModelInput,
        run: RunContext,
    ) -> ChatModelOutput:
        try:
            return await super()._create(input, run)
        except BadRequestError as e:
            if not self._fallback_failed_generation:
                raise e

            match = find_first_pair(e.message, ("{", "}"))
            if not match:
                raise e

            result = parse_broken_json(match.outer, {})
            failed_generation = result.get("error", {}).get("failed_generation")
            if not failed_generation:
                raise e

            return ChatModelOutput(output=[AssistantMessage(failed_generation)])

    @override
    def _format_response_model(self, model: type[BaseModel] | dict[str, Any]) -> type[BaseModel] | dict[str, Any]:
        result = super()._format_response_model(model)
        return self._update_response_format(result)

    @override
    def _format_tool_model(self, model: type[BaseModel]) -> dict[str, Any]:
        result = super()._format_tool_model(model)
        return self._update_response_format(result)

    def _update_response_format(self, model: type[BaseModel] | dict[str, Any]) -> dict[str, Any]:
        """Groq supports just a subset of the JSON Schema."""

        json_schema = model.model_json_schema() if is_pydantic_model(model) else copy.deepcopy(model)  # type: ignore
        json_schema = inline_schema_refs(json_schema)
        simplify_json_schema(
            json_schema,
            SimplifyJsonSchemaConfig(
                excluded_properties_by_type={
                    "array": {"minItems", "maxItems"},
                    "string": {"format", "minLength", "maxLength"},
                }
            ),
        )
        return json_schema

    @override
    async def clone(self) -> Self:
        cloned = await super().clone()
        cloned._fallback_failed_generation = self._fallback_failed_generation
        return cloned
