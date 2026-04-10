import json
import asyncio
from typing import Type, TypeVar, Any
from pydantic import BaseModel, ValidationError

# Ragas Imports
from ragas.llms.base import InstructorBaseRagasLLM

# BeeAI Native Imports - Direct SDK usage without LangChain
from beeai_framework.backend import ChatModel
from beeai_framework.backend.message import UserMessage

# Generic type variable for autocomplete
T = TypeVar("T", bound=BaseModel)

class InstructorRagasLLM(InstructorBaseRagasLLM):
    """
    A class that bridges Ragas with BeeAI directly (without LangChain intermediary).
    """

    def __init__(self, model_name: str):
        # Initialize the BeeAI model directly
        self.model = ChatModel.from_name(model_name)

    async def agenerate(self, prompt: str, response_model: Type[T]) -> T:
        """
        The main function that performs the integration:
        1. Takes a Ragas request.
        2. Converts it to a format BeeAI understands (UserMessage).
        3. Uses BeeAI's native response_format for structured output.
        4. Returns a Pydantic object.
        """
        
        # Create message in BeeAI format
        native_message = UserMessage(prompt)

        # Direct execution with BeeAI engine
        # Use run with response_format instead of generate/invoke
        response = await self.model.run([native_message], response_format=response_model)
        raw_text = response.get_text_content()

        # Clean and parse the JSON
        clean_json = raw_text.strip()
        # Remove Markdown formatting if the model added it
        if clean_json.startswith("```json"): clean_json = clean_json[7:]
        if clean_json.startswith("```"): clean_json = clean_json[3:]
        if clean_json.endswith("```"): clean_json = clean_json[:-3]
        clean_json = clean_json.strip()

        try:
            # Convert to Pydantic object
            return response_model.model_validate_json(clean_json)
        except Exception as e:
            print(f"❌ JSON Parse Error using Native BeeAI. Output:\n{raw_text}")
            raise e

    def generate(self, prompt: str, response_model: Type[T]) -> T:
        """
        Synchronous version (required to implement due to inheritance).
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()

        return loop.run_until_complete(self.agenerate(prompt, response_model))
    
    @staticmethod
    def from_name(model_name: str, **kwargs: Any) -> "InstructorRagasLLM":
        """
        Static factory method to create a InstructorRagasLLM instance from a model name.
        """

        return InstructorRagasLLM(model_name=model_name)