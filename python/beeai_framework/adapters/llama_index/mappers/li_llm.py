from typing import Any, cast

from pydantic import ConfigDict

from beeai_framework.backend import AnyMessage, UserMessage
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.types import ChatModelOutput
from beeai_framework.utils.asynchronous import run_sync

try:
    from llama_index.core.base.llms.types import CompletionResponse, CompletionResponseGen, LLMMetadata, CompletionResponseAsyncGen
    from llama_index.core.llms.callbacks import llm_completion_callback
    from llama_index.core.llms.custom import CustomLLM
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [llama_index] not found.\nRun 'pip install \"beeai-framework[llama_index]\"' to install."
    ) from e


class LlamaIndexChatModel(CustomLLM):
    llm: ChatModel
    
    def __init__(self, llm: ChatModel, *args: Any, **kwargs: Any) -> None:
        super().__init__(llm=llm, *args, **kwargs)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(model_name=self.llm.model_id, is_chat_model=True)

    @llm_completion_callback()
    async def acomplete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        messages: list[AnyMessage] = [UserMessage(prompt)]
        # Formatted argument is neglected as no structure is enforced
        response: ChatModelOutput = await self.llm.create(messages=messages)
        completion_response = CompletionResponse(text=response.messages[-1].text)
        return completion_response
    
    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        result: CompletionResponse = run_sync(self.acomplete(prompt, formatted, **kwargs))
        return result
    
    @llm_completion_callback()
    async def astream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseAsyncGen:
        raise NotImplementedError("Stream completion is not currently supported in LlamaIndex Chat Models")
    
    @llm_completion_callback()
    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseGen:
        return run_sync(self.astream_complete(prompt, formatted, **kwargs))
