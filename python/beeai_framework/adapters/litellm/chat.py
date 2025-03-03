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


import json
import logging
from abc import ABC
from collections.abc import AsyncGenerator
from typing import Any

import litellm
from litellm import (
    ModelResponse,
    ModelResponseStream,
    acompletion,
    get_supported_openai_params,
)
from pydantic import BaseModel, ConfigDict

from beeai_framework.backend.chat import (
    ChatModel,
    ChatModelInput,
    ChatModelOutput,
    ChatModelStructureInput,
    ChatModelStructureOutput,
)
from beeai_framework.backend.errors import ChatModelError
from beeai_framework.backend.message import AssistantMessage, Message, Role, ToolMessage, ToolResult
from beeai_framework.backend.utils import parse_broken_json
from beeai_framework.context import RunContext
from beeai_framework.tools.tool import Tool
from beeai_framework.utils.custom_logger import BeeLogger
from beeai_framework.utils.strings import to_json

logger = BeeLogger(__name__)


class LiteLLMParameters(BaseModel):
    model: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None
    response_format: dict[str, Any] | type[BaseModel] | None = None

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class LiteLLMChatModel(ChatModel, ABC):
    @property
    def model_id(self) -> str:
        return self._model_id

    def __init__(self, model_id: str, *, provider_id: str, settings: dict | None = None) -> None:
        super().__init__()
        self._model_id = model_id
        self._litellm_provider_id = provider_id
        self.supported_params = get_supported_openai_params(model=self.model_id, custom_llm_provider=provider_id) or []
        # drop any unsupported parameters that were passed in
        litellm.drop_params = True
        self._settings = settings or {}

    @staticmethod
    def litellm_debug(enable: bool = True) -> None:
        litellm.set_verbose = enable
        litellm.suppress_debug_info = not enable
        litellm.logging = enable

        logger = logging.getLogger("LiteLLM")
        logger.setLevel(logging.DEBUG if enable else logging.CRITICAL + 1)

    async def _create(
        self,
        input: ChatModelInput,
        run: RunContext,
    ) -> ChatModelOutput:
        litellm_input = self._transform_input(input)
        response = await acompletion(**litellm_input.model_dump())
        response_message = response.get("choices", [{}])[0].get("message", {})
        response_content = response_message.get("content", "")
        tool_calls = response_message.tool_calls

        if tool_calls:
            litellm_input.messages.append({"role": Role.ASSISTANT, "content": response_content})
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call: Tool = next(filter(lambda t: t.name == function_name, input.tools or []))

                function_args = json.loads(tool_call.function.arguments)
                function_response = function_to_call.run(input=function_args)
                litellm_input.messages.append({"role": Role.TOOL, "content": function_response})

                response = await acompletion(**litellm_input.model_dump())

        response_output = self._transform_output(response)
        logger.debug(f"Inference response output:\n{response_output}")
        return response_output

    async def _create_stream(self, input: ChatModelInput, _: RunContext) -> AsyncGenerator[ChatModelOutput]:
        # TODO: handle tool calling for streaming
        litellm_input = self._transform_input(input)
        parameters = litellm_input.model_dump()
        parameters["stream"] = True
        response = await acompletion(**parameters)

        is_empty = True
        async for chunk in response:
            is_empty = False
            response_output = self._transform_output(chunk)
            if response_output:
                yield response_output

        if is_empty:
            # TODO: issue https://github.com/BerriAI/litellm/issues/8868
            raise ChatModelError("Stream response is empty.")

    async def _create_structure(self, input: ChatModelStructureInput, run: RunContext) -> ChatModelStructureOutput:
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

    def _transform_input(self, input: ChatModelInput) -> LiteLLMParameters:
        messages: list[dict] = []
        for message in input.messages:
            if isinstance(message, ToolMessage):
                for chunk in message.content:
                    content = ToolResult.model_validate(chunk)
                    messages.append(
                        {
                            "tool_call_id": content.tool_call_id,
                            "role": "tool",
                            "name": content.tool_name,
                            "content": to_json(content.result),
                        }
                    )
            else:
                messages.append(message.to_plain())

        tools = [{"type": "function", "function": tool.prompt_data()} for tool in input.tools] if input.tools else None

        params = self._settings | self.parameters.model_dump(exclude_unset=True)
        return LiteLLMParameters(
            model=f"{self._litellm_provider_id}/{self.model_id}",
            messages=messages,
            tools=tools,
            response_format=input.response_format,
            **params,
        )

    def _transform_output(self, chunk: ModelResponse | ModelResponseStream) -> ChatModelOutput:
        choice = chunk.get("choices", [{}])[0]
        finish_reason = choice.get("finish_reason")
        message: Message | None = None
        usage = choice.get("usage")

        if isinstance(chunk, ModelResponseStream):
            content = choice.get("delta", {}).get("content")
            if choice.get("tool_calls"):
                message = ToolMessage(content)
            elif choice.get("delta"):
                message = AssistantMessage(content)
            else:
                # TODO: handle other possible types
                raise ChatModelError(f"Unhandled event: {choice}")
        else:
            response_message = choice.get("message")
            content = response_message.get("content")
            message = AssistantMessage(content)

        return ChatModelOutput(messages=[message], finish_reason=finish_reason, usage=usage)


LiteLLMChatModel.litellm_debug(False)
