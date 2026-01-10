from dataclasses import dataclass, field
from typing import List, Optional, Any
import typing as t

# Ragas & LangChain Imports
from ragas.llms.base import BaseRagasLLM
from langchain_core.outputs import LLMResult, Generation
from langchain_core.prompt_values import PromptValue
from langchain_core.callbacks import Callbacks

from beeai_framework.backend import ChatModel
from beeai_framework.backend.message import UserMessage

@dataclass
class RagasLLM(BaseRagasLLM):
    model: Optional["ChatModel"] = field(default=None, repr=False)
    
    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 0.01,
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        # 4. Sync Implementation (Blocking)
        raise NotImplementedError("Sync generation not supported. Use 'agenerate_text'.")
    
    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: Optional[float] = 0.01,
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        """
        Asynchronous generation method (The core logic).
        
        Adapts Ragas input (PromptValue) -> Internal Model -> Ragas output (LLMResult).
        """
       
        # Convert Ragas prompt object to string
        prompt_text = prompt.to_string()
        
        # Wrap string in your internal UserMessage
        input_msg = UserMessage(prompt_text)

        # Run your internal model asynchronously
        response = await self.model.run(
            [input_msg],
            temperature=temperature if temperature is not None else 0.01,
        )

        # Extract text and wrap in LangChain format
        response_text = response.get_text_content()
        generations = [[Generation(text=response_text)]]

        return LLMResult(generations=generations)

    @staticmethod
    def from_name(model_name: str, **kwargs: Any) -> "RagasLLM":
        """
        Static factory method to create a RagasLLM instance from a model name.
        """
        internal_model = ChatModel.from_name(model_name, **kwargs)        
        return RagasLLM(model=internal_model)
    
    def is_finished(self, response: LLMResult) -> bool:
        """
        Validates if the generation process completed successfully and returned data.
        """
        return len(response.generations) > 0 and len(response.generations[0]) > 0
    
    