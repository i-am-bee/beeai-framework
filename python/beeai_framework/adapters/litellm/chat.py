# Copyright 2025 © BeeAI a Series of LF Projects, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from abc import ABC
from collections.abc import AsyncGenerator
from itertools import chain
from typing import Any, Self

from beeai_framework.adapters.litellm.utils import litellm_debug

if not os.getenv("LITELLM_LOCAL_MODEL_COST_MAP", None):
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

import litellm
from litellm import (  # type: ignore
    ModelResponse,
    ModelResponseStream,
    acompletion,
    get_supported_openai_params,
)
from litellm.types.utils import StreamingChoices
from openai.lib._pydantic import _ensure_strict_json_schema, to_strict_json_schema
from pydantic import BaseModel
from typing_extensions import Unpack

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
    ChatModelInput,
    ChatModelOutput,
    ChatModelParameters,
    ChatModelStructureInput,
    ChatModelStructureOutput,
    ChatModelUsage,
)
from beeai_framework.backend.utils import parse_broken_json
from beeai_framework.cache.null_cache import NullCache
from beeai_framework.context import RunContext
from beeai_framework.logger import Logger
from beeai_framework.tools.tool import Tool
from beeai_framework.utils.dicts import exclude_keys, exclude_none, include_keys
from beeai_framework.utils.strings import to_json

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
        litellm.disable_cache()  # type: ignore [attr-defined]

    async def _create(
        self,
        input: ChatModelInput,
        run: RunContext,
    ) -> ChatModelOutput:
        litellm_input = self._transform_input(input) | {"stream": False}
        response = await acompletion(**litellm_input)
        response_output = self._transform_output(response)
        logger.debug(f"Inference response output:\n{response_output}")
        return response_output

    async def _create_stream(self, input: ChatModelInput, _: RunContext) -> AsyncGenerator[ChatModelOutput]:
        litellm_input = self._transform_input(input) | {"stream": True}
        response = await acompletion(**litellm_input)

        is_empty = True
        async for chunk in response:
            is_empty = False
            response_output = self._transform_output(chunk)
            if response_output:
                yield response_output

        if is_empty:
            # TODO: issue https://github.com/BerriAI/litellm/issues/8868
            raise ChatModelError("Stream response is empty.")

    async def _create_structure(self, input: ChatModelStructureInput[Any], run: RunContext) -> ChatModelStructureOutput:
        if "response_format" not in self.supported_params:
            logger.warning(f"{self.provider_id} model {self.model_id} does not support structured data.")
            return await super()._create_structure(input, run)
        else:
            response = await self._create(
                ChatModelInput(
                    messages=input.messages, response_format=input.input_schema, abort_signal=input.abort_signal
                ),
                run,
            )

            logger.debug(f"Structured response received:\n{response}")

            text_response = response.get_text_content()
            result = parse_broken_json(text_response)
            # TODO: validate result matches expected schema
            return ChatModelStructureOutput(object=result)

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
                                {"tool_call_id": content.tool_call_id, "result": content.result}, indent=2
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
                            *[MessageTextContent(text=to_json(t, indent=2)).model_dump() for t in msg_tool_calls],
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
            {*self.supported_params, "abort_signal", "model", "messages", "tools"},
        )
        params = include_keys(
            input.model_dump(exclude_none=True)  # get all parameters with default values
            | self._settings  # get constructor overrides
            | self.parameters.model_dump(exclude_unset=True)  # get default parameters
            | input.model_dump(exclude_none=True, exclude_unset=True),  # get custom manually set parameters
            set(self.supported_params),
        )

        tool_choice = (
            {"type": "function", "function": {"name": input.tool_choice.name}}
            if isinstance(input.tool_choice, Tool)
            else input.tool_choice
        )

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
                "tool_choice": tool_choice if tools else None,
            }
        )

    def _transform_output(self, chunk: ModelResponse | ModelResponseStream) -> ChatModelOutput:
        choice = chunk.choices[0]
        finish_reason = choice.finish_reason
        usage = chunk.get("usage")  # type: ignore
        update = choice.delta if isinstance(choice, StreamingChoices) else choice.message

        return ChatModelOutput(
            messages=(
                [
                    (
                        AssistantMessage(
                            [
                                MessageToolCallContent(
                                    id=call.id or "dummy_id",
                                    tool_name=call.function.name or "",
                                    args=call.function.arguments,
                                )
                                for call in update.tool_calls
                            ]
                        )
                        if update.tool_calls
                        else AssistantMessage(update.content)  # type: ignore
                    )
                ]
                if update.model_dump(exclude_none=True)
                else []
            ),
            finish_reason=finish_reason,
            usage=ChatModelUsage(**usage.model_dump()) if usage else None,
        )

    def _format_tool_model(self, model: type[BaseModel]) -> dict[str, Any]:
        return to_strict_json_schema(model) if self.use_strict_tool_schema else model.model_json_schema()

    def _format_response_model(self, model: type[BaseModel] | dict[str, Any]) -> type[BaseModel] | dict[str, Any]:
        if isinstance(model, dict) and model.get("type") in ["json_schema", "json_object"]:
            return model

        json_schema = (
            {
                "schema": _ensure_strict_json_schema(model, path=(), root=model)
                if self.use_strict_tool_schema
                else model,
                "name": "schema",
                "strict": self.use_strict_model_schema,
            }
            if isinstance(model, dict)
            else {
                "schema": to_strict_json_schema(model) if self.use_strict_tool_schema else model.model_json_schema(),
                "name": model.__name__,
                "strict": self.use_strict_model_schema,
            }
        )

        return {"type": "json_schema", "json_schema": json_schema}

    async def clone(self) -> Self:
        cloned = type(self)(self._model_id, settings=self._settings.copy())  # type: ignore
        cloned.parameters = self.parameters.model_copy() if self.parameters else ChatModelParameters()
        cloned.cache = await self.cache.clone() if self.cache else NullCache[list[ChatModelOutput]]()
        cloned.tool_call_fallback_via_response_format = self.tool_call_fallback_via_response_format
        cloned.model_supports_tool_calling = self.model_supports_tool_calling
        cloned.use_strict_model_schema = self.use_strict_model_schema
        cloned.use_strict_tool_schema = self.use_strict_tool_schema
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
