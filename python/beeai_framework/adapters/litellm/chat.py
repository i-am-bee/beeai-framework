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
from beeai_framework.backend.message import AssistantMessage, Message, Role, ToolMessage
from beeai_framework.backend.utils import parse_broken_json
from beeai_framework.context import RunContext
from beeai_framework.tools.tool import Tool
from beeai_framework.utils.custom_logger import BeeLogger

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

    def __init__(self, model_id: str, **settings: Any) -> None:
        self._model_id = model_id
        llm_provider = "ollama_chat" if self.provider_id == "ollama" else self.provider_id
        self.supported_params = get_supported_openai_params(model=self.model_id, custom_llm_provider=llm_provider) or []
        # drop any unsupported parameters that were passed in
        litellm.drop_params = True
        self.settings = settings
        super().__init__()

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
        litellm_input = self._transform_input(input)
        parameters = litellm_input.model_dump()
        parameters["stream"] = True
        response = await acompletion(**parameters)

        # TODO: handle tool calling for streaming
        async for chunk in response:
            response_output = self._transform_output(chunk)
            if not response_output:
                continue
            yield response_output

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

    def _get_model_name(self) -> str:
        return f"{'ollama_chat' if self.provider_id == 'ollama' else self.provider_id}/{self.model_id}"

    def _transform_input(self, input: ChatModelInput) -> LiteLLMParameters:
        messages_list = [message.to_plain() for message in input.messages]

        if input.tools:
            prepared_tools_list = [{"type": "function", "function": tool.prompt_data()} for tool in input.tools]
        else:
            prepared_tools_list = None

        model = self._get_model_name()

        return LiteLLMParameters(
            model=model,
            messages=messages_list,
            tools=prepared_tools_list,
            response_format=input.response_format,
            **self.settings,
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
