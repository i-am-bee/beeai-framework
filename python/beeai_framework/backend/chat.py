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


from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Callable
from typing import Any, Literal, Self, TypeVar

from pydantic import BaseModel, ConfigDict, Field, InstanceOf

from beeai_framework.backend.constants import ProviderName
from beeai_framework.backend.errors import ChatModelError
from beeai_framework.backend.message import AssistantMessage, Message, MessageToolCallContent, SystemMessage
from beeai_framework.backend.utils import load_model, parse_broken_json, parse_model
from beeai_framework.cancellation import AbortController, AbortSignal
from beeai_framework.context import Run, RunContext, RunContextInput, RunInstance
from beeai_framework.emitter import Emitter
from beeai_framework.logger import Logger
from beeai_framework.retryable import Retryable, RetryableConfig, RetryableContext, RetryableInput
from beeai_framework.template import PromptTemplate, PromptTemplateInput
from beeai_framework.tools.tool import Tool
from beeai_framework.utils.lists import flatten
from beeai_framework.utils.models import ModelLike
from beeai_framework.utils.strings import to_json

T = TypeVar("T", bound=BaseModel)
ChatModelFinishReason: Literal["stop", "length", "function_call", "content_filter", "null"]
logger = Logger(__name__)


class ChatModelParameters(BaseModel):
    max_tokens: int | None = None
    top_p: int | None = None
    frequency_penalty: int | None = None
    temperature: int = 0
    top_k: int | None = None
    n: int | None = None
    presence_penalty: int | None = None
    seed: int | None = None
    stop_sequences: list[str] | None = None
    stream: bool | None = None


class ChatConfig(BaseModel):
    # TODO: cache: ChatModelCache | Callable[[ChatModelCache], ChatModelCache] | None = None
    parameters: ChatModelParameters | Callable[[ChatModelParameters], ChatModelParameters] | None = None


class ChatModelStructureInput(ChatModelParameters):
    input_schema: type[T] = Field(..., alias="schema")
    messages: list[InstanceOf[Message]] = Field(..., min_length=1)
    abort_signal: AbortSignal | None = None
    max_retries: int | None = None


class ChatModelStructureOutput(BaseModel):
    object: type[T] | dict[str, Any]


class ChatModelInput(ChatModelParameters):
    tools: list[InstanceOf[Tool]] | None = None
    abort_signal: AbortSignal | None = None
    stop_sequences: list[str] | None = None
    response_format: dict[str, Any] | type[BaseModel] | None = None
    # tool_choice: NoneType # TODO
    messages: list[InstanceOf[Message]] = Field(
        ...,
        min_length=1,
        frozen=True,
    )

    model_config = ConfigDict(frozen=True)


class ChatModelUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatModelOutput(BaseModel):
    messages: list[InstanceOf[Message]]
    usage: InstanceOf[ChatModelUsage] | None = None
    finish_reason: str | None = None

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
            if other.usage.total_tokens:
                merged_usage.total_tokens = max(self.usage.total_tokens, other.usage.total_tokens)
                merged_usage.prompt_tokens = max(self.usage.prompt_tokens, other.usage.prompt_tokens)
                merged_usage.completion_tokens = max(self.usage.completion_tokens, other.usage.completion_tokens)
            self.usage = merged_usage
        elif other.usage:
            self.usage = other.usage.model_copy()

    def get_tool_calls(self) -> list[MessageToolCallContent]:
        assistant_message = [msg for msg in self.messages if isinstance(msg, AssistantMessage)]
        return flatten([x.get_tool_calls() for x in assistant_message])

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
        self.parameters = ChatModelParameters()
        self.emitter = Emitter.root().child(
            namespace=["backend", self.provider_id, "chat"],
            creator=self,
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
            PromptTemplateInput(
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
        )

        input_messages = input.messages
        messages: list[Message] = [
            SystemMessage(system_template.render({"schema": to_json(json_schema, indent=4)})),
            *input_messages,
        ]

        class DefaultChatModelStructureErrorSchema(BaseModel):
            errors: str
            expected: str
            received: str

        async def executor(_: RetryableContext) -> ChatModelStructureOutput:
            response = await self._create(
                ChatModelInput(
                    messages=messages, response_format={"type": "object-json"}, abort_signal=input.abort_signal
                ),
                run,
            )

            logger.debug(f"Recieved structured response:\n{response}")

            text_response = response.get_text_content()
            result = parse_broken_json(text_response)
            # TODO: validate result matches expected schema
            return ChatModelStructureOutput(object=result)

        return await Retryable(
            RetryableInput(
                executor=executor,
                config=RetryableConfig(max_retries=input.max_retries if input else 1, signal=run.signal),
            )
        ).get()

    def create(
        self,
        *,
        messages: list[Message],
        tools: list[Tool] | None = None,
        abort_signal: AbortSignal | None = None,
        stop_sequences: list[str] | None = None,
        response_format: dict[str, Any] | type[BaseModel] | None = None,
        stream: bool | None = None,
        **kwargs: Any,
    ) -> Run[ChatModelOutput]:
        model_input = ChatModelInput(
            messages=messages,
            tools=tools or [],
            abort_signal=abort_signal,
            stop_sequences=stop_sequences,
            response_format=response_format,
            stream=stream,
            **kwargs,
        )

        async def run_create(context: RunContext) -> ChatModelOutput:
            try:
                await context.emitter.emit("start", {"input": model_input})
                chunks: list[ChatModelOutput] = []

                if model_input.stream:
                    abort_controller: AbortController = AbortController()
                    async for value in self._create_stream(model_input, context):
                        chunks.append(value)
                        await context.emitter.emit(
                            "newToken", {"value": value, "abort": lambda: abort_controller.abort()}
                        )
                        if abort_controller.signal.aborted:
                            break

                    result = ChatModelOutput.from_chunks(chunks)
                else:
                    result = await self._create(model_input, context)

                await context.emitter.emit("success", {"value": result})
                return result
            except Exception as ex:
                error = ChatModelError.ensure(ex)
                await context.emitter.emit("error", {"error": error})
                raise error
            finally:
                await context.emitter.emit("finish", None)

        return RunContext.enter(
            RunInstance(emitter=self.emitter),
            RunContextInput(params=[model_input], signal=model_input.abort_signal),
            run_create,
        )

    def create_structure(
        self,
        *,
        schema: type[T],
        messages: list[Message],
        abort_signal: AbortSignal | None = None,
        max_retries: int | None = None,
    ) -> Run:
        model_input = ChatModelStructureInput(
            schema=schema, messages=messages, abort_signal=abort_signal, max_retries=max_retries
        )

        async def run_structure(context: RunContext) -> ChatModelStructureOutput:
            return await self._create_structure(model_input, context)

        return RunContext.enter(
            RunInstance(emitter=self.emitter),
            RunContextInput(params=[model_input], signal=model_input.abort_signal),
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

        return TargetChatModel(parsed_model.model_id, settings=settings or {})
