# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from beeai_framework.context import RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.tools import JSONToolOutput, Tool, ToolRunOptions


@runtime_checkable
class Reader(Protocol):
    @abstractmethod
    def write(self, prefix: str, message: str) -> None:
        pass

    @abstractmethod
    def ask_single_question(self, prefix: str) -> str:
        pass


class HumanToolOutputResult(BaseModel):
    clarification: str


class HumanToolOutput(JSONToolOutput[HumanToolOutputResult]):
    pass


class HumanToolInput(BaseModel):
    message: str = Field(min_length=1)


class HumanTool(Tool[HumanToolInput, ToolRunOptions, HumanToolOutput]):
    name = "HumanTool"
    description = """
    This tool is used whenever the user's input is unclear, ambiguous, or incomplete.
    The agent MUST invoke this tool when additional clarification is required to proceed.
    The output must adhere strictly to the following structure:
        - Thought: A single-line description of the need for clarification.
        - Function Name: HumanTool
        - Function Input: { "message": "Your question to the user for clarification." }
        - Function Output: The user's response in JSON format.
    Examples:
        - Example 1:
          Input: "What is the weather?"
          Thought: "The user's request lacks a location. I need to ask for clarification."
          Function Name: HumanTool
          Function Input: { "message": "Could you provide the location for which you would like to know the weather?" }
          Function Output: { "clarification": "Santa Fe, Argentina" }
          Final Answer: The current weather in Santa Fe, Argentina is 17.3°C with a relative humidity of 48% and a wind speed of 10.1 km/h.

        - Example 2:
          Input: "Can you help me?"
          Thought: "The user's request is too vague. I need to ask for more details."
          Function Name: HumanTool
          Function Input: { "message": "Could you clarify what kind of help you need?" }
          Function Output: { "clarification": "I need help understanding how to use the project management tool." }
          Final Answer: Sure, I can help you with the project management tool. Let me know which feature you'd like to learn about or if you'd like a general overview.

        - Example 3:
          Input: "Translate this sentence."
          Thought: "The user's request is incomplete. I need to ask for the sentence they want translated."
          Function Name: HumanTool
          Function Input: { "message": "Could you specify the sentence you would like me to translate?" }
          Function Output: { "clarification": "Translate 'Hello, how are you?' to French." }
          Final Answer: The French translation of 'Hello, how are you?' is 'Bonjour, comment vas-tu?'

    Note: Do NOT attempt to guess or provide incomplete responses. Always use this tool when in doubt to ensure accurate and meaningful interactions.
"""

    def __init__(self, *, reader: Reader, name: str | None = None, description: str | None = None) -> None:
        super().__init__()
        self._reader = reader
        self.name = name or self.name
        self.description = description or self.description

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "human"],
            creator=self,
        )

    @property
    def input_schema(self) -> type[HumanToolInput]:
        return HumanToolInput

    async def _run(
        self, tool_input: HumanToolInput, options: ToolRunOptions | None, run: RunContext
    ) -> HumanToolOutput:
        # Use the reader from input
        self._reader.write("HumanTool", tool_input.message)

        # Use ask_single_question
        user_input: str = self._reader.ask_single_question("User 👤 (clarification) : ")

        # Return JSONToolOutput with the clarification
        result = HumanToolOutputResult(clarification=user_input.strip())
        return HumanToolOutput(result)
