# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import copy
import os
from collections.abc import AsyncGenerator
from typing import Any

import litellm.exceptions
from litellm.exceptions import BadRequestError, MidStreamFallbackError
from pydantic import BaseModel
from typing_extensions import Unpack, override

from beeai_framework.adapters.groq.backend._errors import GroqChatModelError
from beeai_framework.adapters.litellm.chat import LiteLLMChatModel
from beeai_framework.backend import AssistantMessage, ChatModelOutput
from beeai_framework.backend.chat import ChatModelKwargs
from beeai_framework.backend.constants import ProviderName
from beeai_framework.backend.errors import ChatModelToolCallError
from beeai_framework.backend.types import ChatModelInput
from beeai_framework.backend.utils import inline_schema_refs
from beeai_framework.context import RunContext
from beeai_framework.logger import Logger
from beeai_framework.utils.models import is_pydantic_model
from beeai_framework.utils.schema import SimplifyJsonSchemaConfig, simplify_json_schema

logger = Logger(__name__)


class GroqChatModel(LiteLLMChatModel):
    @property
    def provider_id(self) -> ProviderName:
        return "groq"

    def __init__(
        self,
        model_id: str | None = None,
        api_key: str | None = None,
        **kwargs: Unpack[ChatModelKwargs],
    ) -> None:
        kwargs.pop("fallback_failed_generation", None)  # type: ignore
        super().__init__(
            model_id if model_id else os.getenv("GROQ_CHAT_MODEL", "llama-3.1-8b-instant"),
            provider_id="groq",
            **kwargs,
        )
        self._assert_setting_value("api_key", api_key, envs=["GROQ_API_KEY"])

    @override
    async def _create(
        self,
        input: ChatModelInput,
        run: RunContext,
    ) -> ChatModelOutput:
        try:
            return await super()._create(input, run)
        except BadRequestError as ex:
            try:
                formated_err = GroqChatModelError.parse_from_litellm_error(ex.message)
            except Exception:
                raise ex

            if "tool" in formated_err.code or "tool" in formated_err.message.lower():
                raise ChatModelToolCallError(
                    generated_error=formated_err.message,
                    generated_content=formated_err.failed_generation or "",
                    context={"source": formated_err},
                )

            return ChatModelOutput(output=[AssistantMessage(formated_err.failed_generation)])

    @override
    # pyrefly: ignore [bad-param-name-override]
    async def _create_stream(self, input: ChatModelInput, ctx: RunContext) -> AsyncGenerator[ChatModelOutput]:
        try:
            async for chunk in super()._create_stream(input, ctx):
                yield chunk
        except MidStreamFallbackError as e:
            source_exc = e.original_exception
            if not isinstance(source_exc, litellm.exceptions.APIError):
                raise e

            # status_code should be int but is actually not
            if "tool" not in str(source_exc.status_code) and "tool" not in str(source_exc.message).lower():
                raise e

            raise ChatModelToolCallError(
                generated_error=source_exc.message.removeprefix("litellm.APIError: APIError: GroqException -").strip(),
                generated_content=e.generated_content,
                context={"source": source_exc},
                cause=e,
            )

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
        # pyrefly: ignore [bad-argument-type]
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
