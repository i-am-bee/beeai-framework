import asyncio
from typing import Any

from pydantic import Field

from beeai_framework.backend import UserMessage
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.types import ChatModelOutput

try:
    from llama_index.core.base.llms.types import CompletionResponse, CompletionResponseGen, LLMMetadata
    from llama_index.core.llms.callbacks import llm_completion_callback
    from llama_index.core.llms.custom import CustomLLM
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [llama_index] not found.\nRun 'pip install \"beeai-framework[llama_index]\"' to install."
    ) from e


def run_in_loop(awaitable: Any) -> Any:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(awaitable)

    # We're inside an already running loop — use `asyncio.run_coroutine_threadsafe`
    # This works only if called from another thread.

    future = asyncio.run_coroutine_threadsafe(awaitable, loop)
    return future.result()


class LlamaIndexLLM(CustomLLM):
    bai_llm: ChatModel = Field(description="The BeeAI LLM instances.", default=None)

    def __init__(self, bai_llm: ChatModel, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.bai_llm = bai_llm

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(model_name=self.bai_llm.model_id, is_chat_model=True)

    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        messages = [UserMessage(prompt)]
        # Formatted argument is neglected as no structure is enforced
        response: ChatModelOutput = run_in_loop(self.bai_llm.create(messages=messages))
        completion_response = CompletionResponse(text=response.messages[-1].text)
        return completion_response

    @llm_completion_callback()
    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseGen:
        raise NotImplementedError("Not implemented")
