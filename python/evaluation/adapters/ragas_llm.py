# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from typing import Any, Type, TypeVar

from pydantic import BaseModel, ValidationError
from ragas.llms.base import InstructorBaseRagasLLM

from beeai_framework.backend import ChatModel
from beeai_framework.backend.message import UserMessage

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class InstructorRagasLLM(InstructorBaseRagasLLM):
    """A class that bridges Ragas with BeeAI directly (without LangChain intermediary)."""

    def __init__(self, model_name: str):
        self.model = ChatModel.from_name(model_name)

    async def agenerate(self, prompt: str, response_model: Type[T]) -> T:
        """
        The main function that performs the integration:
        1. Takes a Ragas request.
        2. Converts it to a format BeeAI understands (UserMessage).
        3. Uses BeeAI's native response_format for structured output.
        4. Returns a Pydantic object.
        """
        native_message = UserMessage(prompt)
        response = await self.model.run([native_message], response_format=response_model)
        raw_text = response.get_text_content()

        clean_json = raw_text.strip()
        if clean_json.startswith("```json"):
            clean_json = clean_json[7:]
        if clean_json.startswith("```"):
            clean_json = clean_json[3:]
        if clean_json.endswith("```"):
            clean_json = clean_json[:-3]
        clean_json = clean_json.strip()

        try:
            return response_model.model_validate_json(clean_json)
        except (ValueError, ValidationError) as e:
            logger.error("JSON Parse Error using Native BeeAI. Output:\n%s", raw_text)
            raise e

    def generate(self, prompt: str, response_model: Type[T]) -> T:
        """Synchronous version (required to implement due to inheritance)."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import nest_asyncio

            nest_asyncio.apply()
            return loop.run_until_complete(self.agenerate(prompt, response_model))

        return asyncio.run(self.agenerate(prompt, response_model))

    @staticmethod
    def from_name(model_name: str, **kwargs: Any) -> "InstructorRagasLLM":
        """Static factory method to create an InstructorRagasLLM instance from a model name."""
        return InstructorRagasLLM(model_name=model_name, **kwargs)
