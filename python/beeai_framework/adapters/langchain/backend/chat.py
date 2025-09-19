# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncGenerator
from typing import Any, Unpack

from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import Runnable as LCRunnable
from pydantic import BaseModel

from beeai_framework.adapters.langchain.backend._utils import beeai_tool_to_lc_tool, to_beeai_messages, to_lc_messages
from beeai_framework.backend import (
    ChatModel,
    ChatModelError,
    ChatModelOutput,
    ChatModelStructureOutput,
)
from beeai_framework.backend.chat import ChatModelKwargs, T
from beeai_framework.backend.constants import ProviderName
from beeai_framework.backend.types import ChatModelInput, ChatModelStructureInput, ChatModelUsage
from beeai_framework.context import RunContext


class LangChainChatModel(ChatModel):
    def __init__(self, model: BaseChatModel, **kwargs: Unpack[ChatModelKwargs]) -> None:
        super().__init__(**kwargs)
        self._model = model

    @property
    def model_id(self) -> str:
        # copied from the LC Base Class
        if hasattr(self._model, "model") and isinstance(self._model, str):
            return self._model.model
        elif hasattr(self._model, "model_name") and isinstance(self._model.model_name, str):
            return self._model.model_name
        else:
            return "unknown"

    @property
    def provider_id(self) -> ProviderName:
        return "langchain"

    def _prepare_model(self, input: ChatModelInput) -> LCRunnable[LanguageModelInput, BaseMessage]:
        parameters = self.parameters.model_dump(exclude_none=True, exclude={"stop_sequences", "stream", "tool_choice"})
        parameters["stop"] = input.stop_sequences

        # Note: setting via bind might not work for certain providers
        for key, value in parameters.items():
            if hasattr(self._model, key):
                setattr(self._model, key, value)

        tools = [beeai_tool_to_lc_tool(tool) for tool in (input.tools or [])]
        tool_choice = str(input.tool_choice) if input.tool_choice else None
        return self._model.bind_tools(tools, tool_choice=tool_choice)

    async def _create(self, input: ChatModelInput, run: RunContext) -> ChatModelOutput:
        input_messages = to_lc_messages(input.messages)
        model = self._prepare_model(input)
        lc_response = await model.ainvoke(input=input_messages, stop=input.stop_sequences)
        return self._transform_output(lc_response)

    async def _create_stream(self, input: ChatModelInput, run: RunContext) -> AsyncGenerator[ChatModelOutput]:
        input_messages = to_lc_messages(input.messages)
        model = self._prepare_model(input)

        tmp_chunk: ChatModelOutput | None = None
        async for _chunk in model.astream(input=input_messages, stop=input.stop_sequences):
            if _chunk is None:
                continue

            chunk = self._transform_output(_chunk)

            if tmp_chunk is None:
                tmp_chunk = chunk
            else:
                tmp_chunk.merge(chunk)

            if tmp_chunk.is_valid():
                yield tmp_chunk
                tmp_chunk = None

        if tmp_chunk:
            raise ChatModelError("Failed to merge intermediate responses.")

    async def _create_structure(self, input: ChatModelStructureInput[T], run: RunContext) -> ChatModelStructureOutput:
        response = await self._model.with_structured_output(schema=input.input_schema, include_raw=True).ainvoke(
            to_lc_messages(input.messages),
            stop=input.stop_sequences,
        )
        assert isinstance(response, dict), "Response must be a dictionary because include_raw was set to True."

        parsing_error = response.get("parsing_error")
        if parsing_error is not None:
            raise ChatModelError("Failed to produce a valid response.") from parsing_error

        parsed_output: dict[str, Any] | None = response.get("parsed")
        if parsed_output is None:
            raise ChatModelError("Failed to produce a valid response. Got an empty result.")

        raw: AIMessage | None = response.get("raw")
        if raw is None:
            raise ChatModelError("Failed to produce a valid response. No message was found.")

        parsed_object = parsed_output.model_dump() if isinstance(parsed_output, BaseModel) else parsed_output
        return ChatModelStructureOutput(object=parsed_object)

    def _transform_output(self, message: BaseMessage) -> ChatModelOutput:
        usage_metadata: dict[str, int] = message.usage_metadata if hasattr(message, "usage_metadata") else {}
        return ChatModelOutput(
            messages=to_beeai_messages([message]),
            finish_reason=message.response_metadata.get("done_reason"),
            usage=ChatModelUsage(
                prompt_tokens=usage_metadata.get("input_tokens") or 0,
                completion_tokens=usage_metadata.get("output_tokens") or 0,
                total_tokens=usage_metadata.get("total_tokens") or 0,
            )
            if usage_metadata
            else None,
        )
