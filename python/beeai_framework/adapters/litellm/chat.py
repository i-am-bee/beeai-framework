# Copyright 2025 IBM Corp.
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

import logging
from abc import ABC
from collections.abc import AsyncGenerator
from typing import Any

import litellm
from litellm import (  # type: ignore
    ModelResponse,
    ModelResponseStream,
    acompletion,
    get_supported_openai_params,
)
from litellm.types.utils import StreamingChoices
from openai.lib._pydantic import to_strict_json_schema
from pydantic import BaseModel

from beeai_framework.backend.chat import (
    ChatModel,
)
from beeai_framework.backend.errors import ChatModelError
from beeai_framework.backend.message import (
    AssistantMessage,
    MessageToolCallContent,
    ToolMessage,
)
from beeai_framework.backend.types import (
    ChatModelInput,
    ChatModelOutput,
    ChatModelStructureInput,
    ChatModelStructureOutput,
    ChatModelUsage,
)
from beeai_framework.backend.utils import parse_broken_json
from beeai_framework.context import RunContext
from beeai_framework.logger import Logger
from beeai_framework.utils.dicts import exclude_keys, exclude_none, include_keys

logger = Logger(__name__)


class LiteLLMChatModel(ChatModel, ABC):
    @property
    def model_id(self) -> str:
        return self._model_id

    def __init__(self, model_id: str, *, provider_id: str, settings: dict[str, Any] | None = None) -> None:
        super().__init__()
        self._model_id = model_id
        self._litellm_provider_id = provider_id
        self.supported_params = get_supported_openai_params(model=self.model_id, custom_llm_provider=provider_id) or []
        # drop any unsupported parameters that were passed in
        litellm.drop_params = True
        self._settings = settings.copy() if settings is not None else {}

    @staticmethod
    def litellm_debug(enable: bool = True) -> None:
        litellm.set_verbose = enable  # type: ignore
        litellm.suppress_debug_info = not enable

        litellm.suppress_debug_info = not enable
        litellm.logging = enable

        logger = logging.getLogger("LiteLLM")
        logger.setLevel(logging.DEBUG if enable else logging.CRITICAL + 1)

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
                    messages.append(
                        {
                            "tool_call_id": content.tool_call_id,
                            "role": "tool",
                            "name": content.tool_name,
                            "content": content.result,
                        }
                    )
            elif isinstance(message, AssistantMessage):
                messages.append(
                    {
                        "role": "assistant",
                        "content": [t.model_dump() for t in message.get_text_messages()] or None,
                        "function_call": None,
                        "tool_calls": [
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
                        or None,
                    }
                )
            else:
                messages.append(message.to_plain())

        tools = (
            [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": self._format_tool_model(tool.input_schema),
                    },
                }
                for tool in input.tools
            ]
            if input.tools
            else None
        )

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

        return (
            exclude_none(settings)
            | exclude_none(params)
            | {
                "model": f"{self._litellm_provider_id}/{self.model_id}",
                "messages": messages,
                "tools": tools,
                "response_format": self._format_response_model(input.response_format)
                if input.response_format
                else None,
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
        # Original OpenAI API requires more strict schema in order to perform well and pass the validation.
        return to_strict_json_schema(model)

    def _format_response_model(self, model: type[BaseModel] | dict[str, Any]) -> type[BaseModel] | dict[str, Any]:
        return model


LiteLLMChatModel.litellm_debug(False)
