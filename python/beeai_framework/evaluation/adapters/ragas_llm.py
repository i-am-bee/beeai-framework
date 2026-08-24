# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from typing import Any, TypeVar

from pydantic import BaseModel

try:
    from ragas.llms.base import InstructorBaseRagasLLM
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [evaluation] not found.\nRun 'pip install \"beeai-framework[evaluation]\"' to install."
    ) from e

from beeai_framework.backend import ChatModel
from beeai_framework.backend.errors import ChatModelError
from beeai_framework.backend.message import UserMessage

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class InstructorRagasLLM(InstructorBaseRagasLLM):
    """A class that bridges Ragas with BeeAI directly (without LangChain intermediary)."""

    def __init__(self, model: ChatModel) -> None:
        self.model = model

    async def agenerate(self, prompt: str, response_model: type[T]) -> T:
        """
        The main function that performs the integration:
        1. Takes a Ragas request.
        2. Converts it to a format BeeAI understands (UserMessage).
        3. Uses BeeAI's native response_format for structured output, which
           already validates and repairs the model's JSON output internally.
        4. Returns a Pydantic object.
        """
        native_message = UserMessage(prompt)
        response = await self.model.run([native_message], response_format=response_model)

        if not isinstance(response.output_structured, response_model):
            raise ChatModelError(
                "The model failed to produce structured output matching the requested schema.",
                context={"output": response.get_text_content()},
            )

        return response.output_structured

    def generate(self, prompt: str, response_model: type[T]) -> T:
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
        model = ChatModel.from_name(model_name, **kwargs)
        return InstructorRagasLLM(model)
