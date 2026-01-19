# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0
import contextlib
import os
from abc import ABC
from collections.abc import AsyncGenerator
from itertools import chain
from typing import Any, Self

from beeai_framework.utils.funcs import safe_invoke

if not os.getenv("LITELLM_LOCAL_MODEL_COST_MAP", None):
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

import litellm
from litellm import (
    ModelResponse,
    ModelResponseStream,
    acompletion,
    cost_per_token,
    get_supported_openai_params,
)
from litellm.types.utils import StreamingChoices
from pydantic import BaseModel
from typing_extensions import Unpack

from beeai_framework.adapters.litellm.utils import (
    fix_double_escaped_tool_calls,
    litellm_debug,
    parse_chat_model_usage,
    process_structured_output,
    to_strict_json_schema,
)
from beeai_framework.backend.chat import (
    ChatModel,
    ChatModelKwargs,
)
from beeai_framework.backend.errors import ChatModelError
from beeai_framework.backend.message import (
    AssistantMessage,
    MessageTextContent,
    MessageToolCallContent,
    ToolMessage,
)
from beeai_framework.backend.types import (
    ChatModelCost,
    ChatModelInput,
    ChatModelOutput,
    ChatModelUsage,
)
from beeai_framework.context import RunContext
from beeai_framework.logger import Logger
from beeai_framework.tools.tool import AnyTool, Tool
from beeai_framework.utils.dicts import exclude_keys, exclude_none, include_keys, set_attr_if_none
from beeai_framework.utils.strings import is_valid_unicode_escape_sequence, to_json

logger = Logger(__name__)


class LiteLLMChatModel(ChatModel, ABC):
    @property
    def model_id(self) -> str:
        return self._model_id

    def __init__(
        self,
        model_id: str,
        *,
        provider_id: str,
        **kwargs: Unpack[ChatModelKwargs],
    ) -> None:
        super().__init__(**kwargs)
        self._model_id = model_id
        self._litellm_provider_id = provider_id
        self.supported_params = get_supported_openai_params(model=self.model_id, custom_llm_provider=provider_id) or []
        # drop any unsupported parameters that were passed in
        litellm.drop_params = True
        # disable LiteLLM caching in favor of our own
        litellm.disable_cache()

    async def _create(
        self,
        input: ChatModelInput,
        run: RunContext,
    ) -> ChatModelOutput:
        litellm_input = self._transform_input(input) | {"stream": False}
        # pyrefly: ignore [not-callable]
        raw = await acompletion(**litellm_input)
        response_output = self._transform_output(raw)
        if not response_output.is_valid():
            fix_double_escaped_tool_calls(response_output.get_tool_calls())

        if not response_output.is_valid():
            raise ChatModelError(
                "Response could not be produced because it is invalid.", context={"output": response_output}
            )

        if input.response_format and not response_output.output_structured:
            text = response_output.get_text_content()
            response_output.output_structured = process_structured_output(
                input.response_format if input.validate_response_format else None, text
            )

        logger.debug(f"Inference response output:\n{response_output}")
        return response_output

    # pyrefly: ignore [bad-param-name-override]
    async def _create_stream(self, input: ChatModelInput, _: RunContext) -> AsyncGenerator[ChatModelOutput]:
        litellm_input = self._transform_input(input) | {"stream": True}
        set_attr_if_none(litellm_input, ["stream_options", "include_usage"], value=True)
        # pyrefly: ignore [not-callable]
        response = await acompletion(**litellm_input)

        text = ""
        is_empty = True
        last_chunk: ChatModelOutput | None = None
        async for _chunk in response:
            is_empty = False
            new_chunk = self._transform_output(_chunk)

            if last_chunk is None:
                last_chunk = new_chunk
            else:
                last_chunk.merge(new_chunk)

            if not is_valid_unicode_escape_sequence(last_chunk.get_text_content()):
                continue

            if input.stream_partial_tool_calls or last_chunk.is_valid():
                text += last_chunk.get_text_content()
                yield last_chunk
                last_chunk = None

        if is_empty:
            # TODO: issue https://github.com/BerriAI/litellm/issues/8868
            raise ChatModelError("Stream response is empty.")

        if last_chunk:
            fix_double_escaped_tool_calls(last_chunk.get_tool_calls())
            if last_chunk.is_valid():
                text += last_chunk.get_text_content()
                yield last_chunk
                last_chunk = None
            else:
                raise ChatModelError(
                    "Response could not be produced because it is invalid.", context={"output": last_chunk}
                )

        if input.response_format:
            output_structured = process_structured_output(
                input.response_format if input.validate_response_format else None, text
            )
            yield ChatModelOutput(output=[], output_structured=output_structured, finish_reason="stop")

    def _transform_input(self, input: ChatModelInput) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        for message in input.messages:
            if isinstance(message, ToolMessage):
                for content in message.content:
                    new_msg = (
                        {
                            "tool_call_id": content.tool_call_id,
                            "role": "tool",
                            "name": content.tool_name,
                            "content": content.result,
                        }
                        if self.model_supports_tool_calling
                        else {
                            "role": "assistant",
                            "content": to_json(
                                {"tool_call_id": content.tool_call_id, "result": content.result},
                                indent=2,
                                sort_keys=False,
                            ),
                        }
                    )
                    messages.append(new_msg)

            elif isinstance(message, AssistantMessage):
                msg_text_content = [t.model_dump() for t in message.get_text_messages()]
                msg_tool_calls = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "arguments": call.args,
                            "name": call.tool_name,
                        },
                    }
                    for call in message.get_tool_calls()
                ]

                new_msg = (
                    {
                        "role": "assistant",
                        "content": msg_text_content or None,
                        "tool_calls": msg_tool_calls or None,
                    }
                    if self.model_supports_tool_calling
                    else {
                        "role": "assistant",
                        "content": [
                            *msg_text_content,
                            *[
                                MessageTextContent(text=to_json(t, indent=2, sort_keys=False)).model_dump()
                                for t in msg_tool_calls
                            ],
                        ]
                        or None,
                    }
                )

                messages.append(exclude_none(new_msg))
            else:
                messages.append(message.to_plain())

        tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": self._format_tool_model(tool.input_schema),
                    "strict": self.use_strict_tool_schema,
                },
            }
            for tool in input.tools or []
        ]

        settings = exclude_keys(
            self._settings | input.model_dump(exclude_unset=True),
            {
                *self.supported_params,
                "signal",
                "model",
                "messages",
                "tools",
                "supports_top_level_unions",
                "validate_response_format",
                "stream_partial_tool_calls",
                "fallback_tool",
            },
        )
        params = include_keys(
            input.model_dump(exclude_none=True)  # get all parameters with default values
            | self._settings  # get constructor overrides
            | self.parameters.model_dump(exclude_unset=True)  # get default parameters
            | input.model_dump(exclude_none=True, exclude_unset=True),  # get custom manually set parameters
            set(self.supported_params),
        )

        tool_choice: dict[str, Any] | str | AnyTool | None = input.tool_choice
        if input.tool_choice == "none" and input.tool_choice not in self._tool_choice_support:
            tool_choice = None
            tools = []
        elif input.tool_choice == "auto" and input.tool_choice not in self._tool_choice_support:
            tool_choice = None
        elif isinstance(input.tool_choice, Tool) and "single" in self._tool_choice_support:
            tool_choice = {"type": "function", "function": {"name": input.tool_choice.name}}
        elif input.tool_choice not in self._tool_choice_support:
            tool_choice = None

        if input.response_format:
            tools = []
            tool_choice = None

        if not self.allow_prompt_caching:
            attr = "cache_control_injection_points"
            params.pop(attr, None)
            settings.pop(attr, None)

        return exclude_none(
            exclude_none(settings)
            | exclude_none(params)
            | {
                "model": f"{self._litellm_provider_id}/{self.model_id}",
                "messages": messages,
                "tools": tools if tools else None,
                "response_format": self._format_response_model(input.response_format)
                if input.response_format
                else None,
                "max_retries": 0,
                "tool_choice": tool_choice if tools else None,
                "parallel_tool_calls": bool(input.parallel_tool_calls) if tools else None,
            }
        )

    def _transform_output(self, chunk: ModelResponse | ModelResponseStream) -> ChatModelOutput:
        model = chunk.get("model")
        usage = chunk.get("usage")
        choice = chunk.choices[0] if chunk.choices else None
        finish_reason = choice.finish_reason if choice else None
        update = (choice.delta if isinstance(choice, StreamingChoices) else choice.message) if choice else None

        cost: ChatModelCost = ChatModelCost()
        with contextlib.suppress(Exception):
            if usage:
                prompt_tokens_cost_usd, completion_tokens_cost_usd = cost_per_token(
                    model=model,
                    custom_llm_provider=self._litellm_provider_id,
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                )
                cost = ChatModelCost(
                    prompt_tokens_usd=prompt_tokens_cost_usd,
                    completion_tokens_cost_usd=completion_tokens_cost_usd,
                    total_cost_usd=prompt_tokens_cost_usd + completion_tokens_cost_usd,
                )

        return ChatModelOutput(
            output=(
                [
                    AssistantMessage(
                        [
                            MessageToolCallContent(
                                id=call.id or "",
                                tool_name=call.function.name or "",
                                args=call.function.arguments,
                            )
                            for call in update.tool_calls
                        ],
                        id=chunk.id,
                    )
                    if update.tool_calls
                    # pyrefly: ignore [bad-argument-type]
                    else AssistantMessage(update.content, id=chunk.id)
                ]
                if (update and update.model_dump(exclude_none=True))
                else []
            ),
            # Will be set later
            output_structured=None,
            finish_reason=finish_reason,
            usage=parse_chat_model_usage(usage) if usage else ChatModelUsage(),
            cost=cost,
        )

    def _format_tool_model(self, model: type[BaseModel]) -> dict[str, Any]:
        return to_strict_json_schema(model) if self.use_strict_tool_schema else model.model_json_schema()

    def _format_response_model(self, model: type[BaseModel] | dict[str, Any]) -> type[BaseModel] | dict[str, Any]:
        if isinstance(model, dict) and model.get("type") in ["json_schema", "json_object"]:
            return model

        strict = self.use_strict_model_schema

        json_schema = (
            {
                "schema": to_strict_json_schema(model) if strict else model,
                "name": "schema",
                "strict": strict,
            }
            if isinstance(model, dict)
            else {
                "schema": to_strict_json_schema(model) if strict else model.model_json_schema(),
                "name": model.__name__,
                "strict": strict,
            }
        )

        return {"type": "json_schema", "json_schema": json_schema}

    async def clone(self) -> Self:
        cloned: Self = safe_invoke(self.__class__)(
            model_id=self.model_id,
            provider_id=self._litellm_provider_id,
            parameters=self.parameters.model_copy(),
            cache=await self.cache.clone() if self.cache else None,
            tool_call_fallback_via_response_format=self.tool_call_fallback_via_response_format,
            model_supports_tool_calling=self.model_supports_tool_calling,
            use_strict_model_schema=self.use_strict_model_schema,
            use_strict_tool_schema=self.use_strict_tool_schema,
            middlewares=self.middlewares.copy(),
            tool_choice_support=self.tool_choice_support.copy(),
            settings=self._settings.copy(),
            allow_parallel_tool_calls=self.allow_parallel_tool_calls,
            ignore_parallel_tool_calls=self.ignore_parallel_tool_calls,
            supports_top_level_unions=self.supports_top_level_unions,
            fix_invalid_tool_calls=self.fix_invalid_tool_calls,
            retry_on_empty_response=self.retry_on_empty_response,
            **self._settings,
        )
        return cloned

    def _assert_setting_value(
        self,
        name: str,
        value: Any | None = None,
        *,
        display_name: str | None = None,
        aliases: list[str] | None = None,
        envs: list[str],
        fallback: str | None = None,
        allow_empty: bool = False,
    ) -> None:
        aliases = aliases or []
        assert aliases is not None

        value = value or self._settings.get(name)
        if not value:
            value = next(
                chain(
                    (self._settings[alias] for alias in aliases if self._settings.get(alias)),
                    (os.environ[env] for env in envs if os.environ.get(env)),
                ),
                fallback,
            )

        for alias in aliases:
            self._settings[alias] = None

        if not value and not allow_empty:
            raise ValueError(
                f"Setting {display_name or name} is required for {type(self).__name__}. "
                f"Either pass the {display_name or name} explicitly or set one of the "
                f"following environment variables: {', '.join(envs)}."
            )

        self._settings[name] = value or None


litellm_debug(False)
