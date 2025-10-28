"""Module for chat model adapter for LlamaIndex."""
# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from beeai_framework.backend import AnyMessage, UserMessage
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.types import ChatModelOutput
from beeai_framework.utils.asynchronous import run_sync

try:
    from llama_index.core.base.llms.types import (
        CompletionResponse,
        CompletionResponseAsyncGen,
        CompletionResponseGen,
        LLMMetadata,
    )
    from llama_index.core.llms.callbacks import llm_completion_callback
    from llama_index.core.llms.custom import CustomLLM
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [llama_index] not found.\nRun 'pip install \"beeai-framework[llama_index]\"' to install."
    ) from e


class LlamaIndexChatModel(CustomLLM):
    """LlamaIndex Chat Model Adapter."""
    llm: ChatModel

    def __init__(self, llm: ChatModel, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, llm=llm, **kwargs)

    @property
    def metadata(self) -> LLMMetadata:
        """Get metadata about the underlying language model.

        Returns:
            LLMMetadata containing the model name and chat model flag.
        """
        return LLMMetadata(model_name=self.llm.model_id, is_chat_model=True)

    @llm_completion_callback()
    async def acomplete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        """Asynchronously generate a completion for the given prompt.

        Args:
            prompt: The input prompt text.
            formatted: Format flag (currently ignored as no structure is enforced).
            **kwargs: Additional keyword arguments (currently unused).

        Returns:
            CompletionResponse containing the generated text.
        """
        messages: list[AnyMessage] = [UserMessage(prompt)]
        # Formatted argument is neglected as no structure is enforced
        response: ChatModelOutput = await self.llm.run(messages)
        completion_response = CompletionResponse(text=response.output[-1].text)
        return completion_response

    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        """Synchronously generate a completion for the given prompt."""
        result: CompletionResponse = run_sync(self.acomplete(prompt, formatted, **kwargs))
        return result

    @llm_completion_callback()
    async def astream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseAsyncGen:
        """Asynchronously stream a completion for the given prompt."""
        raise NotImplementedError("Stream completion is not currently supported in LlamaIndex Chat Models")

    @llm_completion_callback()
    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseGen:
        """Synchronously stream a completion for the given prompt."""
        raise NotImplementedError("Stream completion is not currently supported in LlamaIndex Chat Models")
