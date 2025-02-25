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
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Callable
from typing import Annotated, Any, Literal, Self, TypeVar

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError

from beeai_framework.backend.constants import ProviderName
from beeai_framework.backend.errors import ChatModelError
from beeai_framework.backend.message import AssistantMessage, Message, SystemMessage
from beeai_framework.backend.utils import load_model, parse_broken_json, parse_model
from beeai_framework.cancellation import AbortController, AbortSignal
from beeai_framework.context import Run, RunContext, RunContextInput, RunInstance
from beeai_framework.emitter import Emitter, EmitterInput
from beeai_framework.tools.tool import Tool
from beeai_framework.utils.custom_logger import BeeLogger
from beeai_framework.utils.models import ModelLike, to_model
from beeai_framework.utils.templates import PromptTemplate

T = TypeVar("T", bound=BaseModel)
ChatModelFinishReason: Literal["stop", "length", "function_call", "content_filter", "null"]
logger = BeeLogger(__name__)


def message_validator(messages: list[Message]) -> list[Message]:
    if len(messages) and not isinstance(messages[0], Message):
        raise ValidationError("incoming data must be a Message")
    return messages


def tool_validator(tools: list[Tool]) -> list[Tool]:
    if len(tools) and not isinstance(tools[0], Tool):
        raise ValidationError("incoming data must be a Tool")
    return tools


class ChatModelParameters(BaseModel):
    max_tokens: int | None = None
    top_p: int | None = None
    frequency_penalty: int | None = None
    temperature: int | None = None
    top_k: int | None = None
    n: int | None = None
    presence_penalty: int | None = None
    seed: int | None = None
    stop_sequences: list[str] | None = None
    stream: bool | None = None


class ChatConfig(BaseModel):
    # TODO: cache: ChatModelCache | Callable[[ChatModelCache], ChatModelCache] | None = None
    parameters: ChatModelParameters | Callable[[ChatModelParameters], ChatModelParameters] | None = None


class ChatModelStructureInput(BaseModel):
    input_schema: type[T] = Field(..., alias="schema")
    messages: Annotated[list, BeforeValidator(message_validator)]
    abort_signal: AbortSignal | None = None
    max_retries: int | None = None


class ChatModelStructureOutput(BaseModel):
    object: type[T] | dict[str, Any]


class ChatModelInput(ChatModelParameters):
    tools: Annotated[list, BeforeValidator(tool_validator)] | None = None
    abort_signal: AbortSignal | None = None
    stop_sequences: list[str] | None = None
    response_format: dict[str, Any] | type[BaseModel] = None
    # tool_choice: NoneType # TODO
    messages: Annotated[list, BeforeValidator(message_validator)]

    model_config = ConfigDict(frozen=True)


class ChatModelUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatModelOutput:
    def __init__(
        self,
        *,
        messages: list[Message],
        usage: ChatModelUsage | None = None,
        finish_reason: str | None = None,
    ) -> None:
        self.messages = messages
        self.usage = usage
        self.finish_reason = finish_reason

    @classmethod
    def from_chunks(cls, chunks: list) -> Self:
        final = cls(messages=[])
        for cur in chunks:
            final.merge(cur)
        return final

    def merge(self, other: Self) -> None:
        self.messages.extend(other.messages)
        self.finish_reason = other.finish_reason
        if self.usage and other.usage:
            merged_usage = self.usage.model_copy()
            if other.usage.get("total_tokens"):
                merged_usage.total_tokens = max(self.usage.total_tokens, other.usage.total_tokens)
                merged_usage.prompt_tokens = max(self.usage.prompt_tokens, other.usage.prompt_tokens)
                merged_usage.completion_tokens = max(self.usage.completion_tokens, other.usage.completion_tokens)
            self.usage = merged_usage
        elif other.usage:
            self.usage = other.usage.model_copy()

    def get_text_content(self) -> str:
        return "".join([x.text for x in list(filter(lambda x: isinstance(x, AssistantMessage), self.messages))])


class ChatModel(ABC):
    emitter: Emitter
    parameters: ChatModelParameters

    @property
    @abstractmethod
    def model_id(self) -> str:
        pass

    @property
    @abstractmethod
    def provider_id(self) -> ProviderName:
        pass

    def __init__(self) -> None:
        self.emitter = Emitter.root().child(
            EmitterInput(
                namespace=["backend", self.provider_id, "chat"],
                creator=self,
            )
        )

    @abstractmethod
    async def _create(
        self,
        input: ChatModelInput,
        run: RunContext,
    ) -> ChatModelOutput:
        raise NotImplementedError

    @abstractmethod
    def _create_stream(
        self,
        input: ChatModelInput,
        run: RunContext,
    ) -> AsyncGenerator[ChatModelOutput]:
        raise NotImplementedError

    @abstractmethod
    async def _create_structure(
        self,
        input: ChatModelStructureInput,
        run: RunContext,
    ) -> ChatModelStructureOutput:
        schema = input.input_schema

        json_schema = schema.model_json_schema(mode="serialization") if issubclass(schema, BaseModel) else schema

        class DefaultChatModelStructureSchema(BaseModel):
            schema: str

        system_template = PromptTemplate(
            schema=DefaultChatModelStructureSchema,
            template=(
                """You are a helpful assistant that generates only valid JSON """
                """adhering to the following JSON Schema.
```
{{schema}}
```
IMPORTANT: You MUST answer with a JSON object that matches the JSON schema above."""
            ),
        )

        input_messages = input.messages
        messages: list[Message] = [
            SystemMessage(system_template.render({"schema": json.dumps(json_schema)})),
            *input_messages,
        ]

        response = await self._create(
            ChatModelInput(messages=messages, response_format={"type": "object-json"}, abort_signal=input.abort_signal),
        )

        logger.debug(f"Recieved structured response:\n{response}")

        text_response = response.get_text_content()
        result = parse_broken_json(text_response)
        # TODO: validate result matches expected schema
        return ChatModelStructureOutput(object=result)

    def create(self, chat_model_input: ModelLike[ChatModelInput]) -> Run:
        input = to_model(ChatModelInput, chat_model_input)

        async def run_create(context: RunContext) -> ChatModelOutput:
            try:
                await context.emitter.emit("start", input)
                chunks: list[ChatModelOutput] = []

                if input.stream:
                    abort_controller: AbortController = AbortController()
                    generator = self._create_stream(input, context)
                    async for value in generator:
                        chunks.append(value)
                        await context.emitter.emit("newToken", (value, lambda: abort_controller.abort()))
                        if abort_controller.signal.aborted:
                            break

                    result = ChatModelOutput.from_chunks(chunks)
                else:
                    result = await self._create(input, context)

                await context.emitter.emit("success", {"value": result})
                return result
            except ChatModelError as error:
                await context.emitter.emit("error", {input, error})
                raise error
            except Exception as ex:
                await context.emitter.emit("error", {input, ex})
                raise ChatModelError("Model error has occurred.") from ex
            finally:
                await context.emitter.emit("finish", None)

        return RunContext.enter(
            RunInstance(emitter=self.emitter),
            RunContextInput(params=[input], signal=input.abort_signal),
            run_create,
        )

    def create_structure(self, structure_input: ModelLike[ChatModelStructureInput]) -> Run:
        input = to_model(ChatModelStructureInput, structure_input)

        async def run_structure(context: RunContext) -> ChatModelStructureOutput:
            return await self._create_structure(input, context)

        return RunContext.enter(
            RunInstance(emitter=self.emitter),
            RunContextInput(params=[input], signal=input.abort_signal),
            run_structure,
        )

    def config(self, chat_config: ChatConfig) -> None:
        # TODO: uncomment when cache is supported/implemented
        # if chat_config.cache:
        #     self.cache = chat_config.cache(self.cache) if callable(chat_config.cache) else  chat_config.cache

        if chat_config.parameters:
            self.parameters = (
                chat_config.parameters(self.parameters) if callable(chat_config.parameters) else chat_config.parameters
            )

    @staticmethod
    def from_name(name: str | ProviderName, options: ModelLike[ChatModelParameters] | None = None) -> "ChatModel":
        parsed_model = parse_model(name)
        TargetChatModel = load_model(parsed_model.provider_id, "chat")  # noqa: N806

        settings = options.model_dump() if isinstance(options, ChatModelParameters) else options

        return TargetChatModel(parsed_model.model_id, **(settings or {}))
