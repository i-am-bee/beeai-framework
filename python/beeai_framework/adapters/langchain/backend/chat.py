# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncGenerator
from typing import Any, TypedDict, Unpack

from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable as LCRunnable
from langchain_core.runnables import RunnableLambda

from beeai_framework.adapters.langchain.backend._utils import beeai_tool_to_lc_tool, to_beeai_messages, to_lc_messages
from beeai_framework.backend import ChatModel, ChatModelError, ChatModelOutput
from beeai_framework.backend.chat import ChatModelKwargs
from beeai_framework.backend.constants import ProviderName
from beeai_framework.backend.types import ChatModelInput, ChatModelUsage
from beeai_framework.context import RunContext

__all__ = ["LangChainChatModel"]


class LCModelResponse(TypedDict):
    output: BaseMessage
    output_structured: Any | None


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

    def _prepare_model(self, input: ChatModelInput) -> LCRunnable[LanguageModelInput, LCModelResponse]:
        parameters = self.parameters.model_dump(exclude_none=True, exclude={"stop_sequences", "stream", "tool_choice"})
        parameters["stop"] = input.stop_sequences

        # Note: setting via bind might not work for certain providers
        for key, value in parameters.items():
            if hasattr(self._model, key):
                setattr(self._model, key, value)

        if input.tools:
            tools = [beeai_tool_to_lc_tool(tool) for tool in (input.tools or [])]
            tool_choice = str(input.tool_choice) if input.tool_choice else None
            return self._model.bind_tools(tools, tool_choice=tool_choice) | RunnableLambda(
                lambda data: LCModelResponse(output=data, output_structured=None)
            )
        elif input.response_format:
            return self._model.with_structured_output(schema=input.response_format, include_raw=True) | RunnableLambda(
                lambda data: LCModelResponse(output=data["raw"], output_structured=data["parsed"])
            )
        else:
            return self._model | RunnableLambda(lambda data: LCModelResponse(output=data, output_structured=None))

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

    def _transform_output(self, response: LCModelResponse) -> ChatModelOutput:
        message = response["output"]
        usage_metadata: dict[str, int] = message.usage_metadata if hasattr(message, "usage_metadata") else {}
        return ChatModelOutput(
            output=to_beeai_messages([message]),
            output_structured=response["output_structured"],
            finish_reason=message.response_metadata.get("done_reason"),
            usage=ChatModelUsage(
                prompt_tokens=usage_metadata.get("input_tokens") or 0,
                completion_tokens=usage_metadata.get("output_tokens") or 0,
                total_tokens=usage_metadata.get("total_tokens") or 0,
            )
            if usage_metadata
            else None,
        )
