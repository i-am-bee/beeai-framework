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
from functools import cached_property
from typing import Any, ClassVar, Literal, Self, TypeVar

from pydantic import BaseModel, ConfigDict, Field, InstanceOf, TypeAdapter
from typing_extensions import TypedDict, Unpack

from beeai_framework.backend.constants import ProviderName
from beeai_framework.backend.errors import ChatModelError
from beeai_framework.backend.events import (
    ChatModelErrorEvent,
    ChatModelNewTokenEvent,
    ChatModelStartEvent,
    ChatModelSuccessEvent,
    chat_model_event_types,
)
from beeai_framework.backend.message import AnyMessage, MessageToolCallContent, SystemMessage
from beeai_framework.backend.types import (
    ChatModelCache,
    ChatModelInput,
    ChatModelOutput,
    ChatModelParameters,
    ChatModelStructureInput,
    ChatModelStructureOutput,
    ChatModelToolChoice,
)
from beeai_framework.backend.utils import (
    filter_tools_by_tool_choice,
    generate_tool_union_schema,
    load_model,
    parse_broken_json,
    parse_model,
)
from beeai_framework.cache.null_cache import NullCache
from beeai_framework.context import Run, RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.logger import Logger
from beeai_framework.retryable import Retryable, RetryableConfig, RetryableContext, RetryableInput
from beeai_framework.template import PromptTemplate, PromptTemplateInput
from beeai_framework.tools.tool import AnyTool, Tool
from beeai_framework.utils import AbortController, AbortSignal
from beeai_framework.utils.asynchronous import to_async_generator
from beeai_framework.utils.models import ModelLike
from beeai_framework.utils.strings import generate_random_string, to_json

T = TypeVar("T", bound=BaseModel)
TTool = TypeVar("TTool", bound=AnyTool)
ChatModelFinishReason: Literal["stop", "length", "function_call", "content_filter", "null"]
logger = Logger(__name__)


class ChatModelKwargs(TypedDict, total=False):
    tool_call_fallback_via_response_format: bool
    model_supports_tool_calling: bool
    parameters: InstanceOf[ChatModelParameters]
    cache: InstanceOf[ChatModelCache]

    __pydantic_config__ = ConfigDict(extra="forbid")  # type: ignore


_ChatModelKwargsAdapter = TypeAdapter(ChatModelKwargs)


class ChatModel(ABC):
    tool_choice_support: ClassVar[set[str]] = {"required", "none", "single", "auto"}
    tool_call_fallback_via_response_format: bool
    model_supports_tool_calling: bool

    @property
    @abstractmethod
    def model_id(self) -> str:
        pass

    @property
    @abstractmethod
    def provider_id(self) -> ProviderName:
        pass

    def __init__(self, **kwargs: Unpack[ChatModelKwargs]) -> None:
        kwargs = _ChatModelKwargsAdapter.validate_python(kwargs)
        self.parameters = kwargs.get("parameters", ChatModelParameters())
        self.cache = kwargs.get("cache", NullCache[list[ChatModelOutput]]())
        self.tool_call_fallback_via_response_format = kwargs.get("tool_call_fallback_via_response_format", True)
        self.model_supports_tool_calling = kwargs.get("model_supports_tool_calling", True)

    @cached_property
    def emitter(self) -> Emitter:
        return self._create_emitter()

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["backend", self.provider_id, "chat"],
            creator=self,
            events=chat_model_event_types,
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
        input: ChatModelStructureInput[T],
        run: RunContext,
    ) -> ChatModelStructureOutput:
        json_schema: dict[str, Any] = (
            input.input_schema
            if isinstance(input.input_schema, dict)
            else input.input_schema.model_json_schema(mode="serialization")
        )

        class DefaultChatModelStructureSchema(BaseModel):
            input_schema: type[str] = Field(..., alias="schema")

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
        messages: list[AnyMessage] = [
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
                config=RetryableConfig(
                    max_retries=input.max_retries if input is not None and input.max_retries is not None else 1,
                    signal=run.signal,
                ),
            )
        ).get()

    def create(
        self,
        *,
        messages: list[AnyMessage],
        tools: list[AnyTool] | None = None,
        tool_choice: ChatModelToolChoice | None = None,
        abort_signal: AbortSignal | None = None,
        stop_sequences: list[str] | None = None,
        response_format: dict[str, Any] | type[BaseModel] | None = None,
        stream: bool | None = None,
        **kwargs: Any,
    ) -> Run[ChatModelOutput]:
        force_tool_call_via_response_format = self._force_tool_call_via_response_format(
            tool_choice=tool_choice,
            tools=tools or [],
            has_custom_response_format=bool(response_format),
        )

        model_input = ChatModelInput(
            messages=messages,
            tools=tools if self.model_supports_tool_calling else None,
            tool_choice=tool_choice,
            abort_signal=abort_signal,
            stop_sequences=stop_sequences,
            response_format=(
                generate_tool_union_schema(filter_tools_by_tool_choice(tools, tool_choice))
                if force_tool_call_via_response_format and tools
                else response_format
            ),
            stream=stream,
            **kwargs,
        )

        async def handler(context: RunContext) -> ChatModelOutput:
            cache_key = self.cache.generate_key(model_input, {"messages": [m.to_plain() for m in model_input.messages]})
            cache_hit = await self.cache.get(cache_key)

            try:
                await context.emitter.emit("start", ChatModelStartEvent(input=model_input))
                chunks: list[ChatModelOutput] = []

                if model_input.stream:
                    generator = (
                        to_async_generator(cache_hit) if cache_hit else self._create_stream(model_input, context)
                    )
                    abort_controller: AbortController = AbortController()
                    async for value in generator:
                        chunks.append(value)
                        await context.emitter.emit(
                            "new_token", ChatModelNewTokenEvent(value=value, abort=lambda: abort_controller.abort())
                        )
                        if abort_controller.signal.aborted:
                            break

                    if not cache_hit:
                        await self.cache.set(cache_key, chunks)
                    result = ChatModelOutput.from_chunks(chunks)
                else:
                    if cache_hit:
                        result = cache_hit[0].model_copy()
                    else:
                        result = await self._create(model_input, context)
                        await self.cache.set(cache_key, [result])

                if force_tool_call_via_response_format and not result.get_tool_calls():
                    msg = result.messages[-1]
                    tool_call: dict[str, Any] = parse_broken_json(msg.text)
                    if not tool_call:
                        raise ChatModelError(f"Failed to produce a valid tool call. Generated output: '{msg.text}'")

                    tool_call_content = MessageToolCallContent(
                        id=f"call_{generate_random_string(8).lower()}",
                        tool_name=tool_call["name"],
                        args=json.dumps(tool_call["parameters"]),
                    )
                    msg.content.clear()
                    msg.content.append(tool_call_content)

                await context.emitter.emit("success", ChatModelSuccessEvent(value=result))
                return result
            except Exception as ex:
                error = ChatModelError.ensure(ex, model=self)
                if cache_hit:
                    await self.cache.delete(cache_key)
                await context.emitter.emit("error", ChatModelErrorEvent(input=model_input, error=error))
                raise error
            finally:
                await context.emitter.emit("finish", None)

        return RunContext.enter(
            self,
            handler,
            signal=abort_signal,
            run_params=model_input.model_dump(),
        )

    def create_structure(
        self,
        *,
        schema: type[T] | dict[str, Any],
        messages: list[AnyMessage],
        abort_signal: AbortSignal | None = None,
        max_retries: int | None = None,
    ) -> Run[ChatModelStructureOutput]:
        model_input = ChatModelStructureInput[T](
            schema=schema, messages=messages, abort_signal=abort_signal, max_retries=max_retries
        )

        async def handler(context: RunContext) -> ChatModelStructureOutput:
            return await self._create_structure(model_input, context)

        return RunContext.enter(
            self,
            handler,
            signal=abort_signal,
            run_params=model_input.model_dump(),
        )

    def config(
        self,
        *,
        parameters: ChatModelParameters | Callable[[ChatModelParameters], ChatModelParameters] | None = None,
        cache: ChatModelCache | Callable[[ChatModelCache], ChatModelCache] | None = None,
    ) -> None:
        if cache is not None:
            self.cache = cache(self.cache) if callable(cache) else cache

        if parameters is not None:
            self.parameters = parameters(self.parameters) if callable(parameters) else parameters

    @staticmethod
    def from_name(
        name: str | ProviderName, options: ModelLike[ChatModelParameters] | None = None, **kwargs: Any
    ) -> "ChatModel":
        parsed_model = parse_model(name)
        TargetChatModel = load_model(parsed_model.provider_id, "chat")  # type: ignore # noqa: N806

        settings = options.model_dump() if isinstance(options, ChatModelParameters) else options

        return TargetChatModel(parsed_model.model_id, settings=settings or {}, **kwargs)  # type: ignore

    def _force_tool_call_via_response_format(
        self,
        *,
        tool_choice: ChatModelToolChoice | None,
        tools: list[AnyTool],
        has_custom_response_format: bool,
    ) -> bool:
        if (
            not tools
            or tool_choice == "none"
            or tool_choice == "auto"
            or tool_choice is None
            or has_custom_response_format
            or not self.tool_call_fallback_via_response_format
        ):
            return False

        tool_choice_supported = not tool_choice or (
            "single" in self.tool_choice_support
            if isinstance(tool_choice, Tool)
            else tool_choice in self.tool_choice_support
        )

        return not self.model_supports_tool_calling or not tool_choice_supported

    async def clone(self) -> Self:
        kwargs: ChatModelKwargs = {
            "parameters": ChatModelParameters(**self.parameters.model_dump())
            if self.parameters
            else ChatModelParameters(),
            "cache": await self.cache.clone() if self.cache else NullCache[list[ChatModelOutput]](),
            "tool_call_fallback_via_response_format": self.tool_call_fallback_via_response_format,
            "model_supports_tool_calling": self.model_supports_tool_calling,
        }
        cloned = type(self)(**kwargs)
        return cloned
