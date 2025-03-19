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
from functools import cached_property
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, Field

from beeai_framework.backend.constants import ProviderName
from beeai_framework.backend.errors import ChatModelError
from beeai_framework.backend.events import (
    ChatModelErrorEvent,
    ChatModelNewTokenEvent,
    ChatModelStartEvent,
    ChatModelSuccessEvent,
    chat_model_event_types,
)
from beeai_framework.backend.message import AnyMessage, SystemMessage
from beeai_framework.backend.types import (
    ChatModelInput,
    ChatModelOutput,
    ChatModelParameters,
    ChatModelStructureInput,
    ChatModelStructureOutput,
)
from beeai_framework.backend.utils import load_model, parse_broken_json, parse_model
from beeai_framework.cancellation import AbortController, AbortSignal
from beeai_framework.context import Run, RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.logger import Logger
from beeai_framework.retryable import Retryable, RetryableConfig, RetryableContext, RetryableInput
from beeai_framework.template import PromptTemplate, PromptTemplateInput
from beeai_framework.tools.tool import AnyTool
from beeai_framework.utils.models import ModelLike
from beeai_framework.utils.strings import to_json

T = TypeVar("T", bound=BaseModel)
ChatModelFinishReason: Literal["stop", "length", "function_call", "content_filter", "null"]
logger = Logger(__name__)


class ChatModel(ABC):
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
        schema: type[T] = input.input_schema

        json_schema = schema.model_json_schema(mode="serialization") if issubclass(schema, BaseModel) else schema

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

        async def handler(context: RunContext) -> ChatModelOutput:
            try:
                await context.emitter.emit("start", ChatModelStartEvent(input=model_input))
                chunks: list[ChatModelOutput] = []

                if model_input.stream:
                    abort_controller: AbortController = AbortController()
                    async for value in self._create_stream(model_input, context):
                        chunks.append(value)
                        await context.emitter.emit(
                            "new_token", ChatModelNewTokenEvent(value=value, abort=lambda: abort_controller.abort())
                        )
                        if abort_controller.signal.aborted:
                            break

                    result = ChatModelOutput.from_chunks(chunks)
                else:
                    result = await self._create(model_input, context)

                await context.emitter.emit("success", ChatModelSuccessEvent(value=result))
                return result
            except Exception as ex:
                error = ChatModelError.ensure(ex, model=self)
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
        schema: type[T],
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
        # TODO: cache: ChatModelCache | Callable[[ChatModelCache], ChatModelCache] | None = None
    ) -> None:
        # TODO: uncomment when cache is supported/implemented
        # if chat_config.cache:
        #     self.cache = chat_config.cache(self.cache) if callable(chat_config.cache) else  chat_config.cache

        if parameters is not None:
            self.parameters = parameters(self.parameters) if callable(parameters) else parameters

    @staticmethod
    def from_name(name: str | ProviderName, options: ModelLike[ChatModelParameters] | None = None) -> "ChatModel":
        parsed_model = parse_model(name)
        TargetChatModel = load_model(parsed_model.provider_id, "chat")  # type: ignore # noqa: N806

        settings = options.model_dump() if isinstance(options, ChatModelParameters) else options

        return TargetChatModel(parsed_model.model_id, settings=settings or {})  # type: ignore
