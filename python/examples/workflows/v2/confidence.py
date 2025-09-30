# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio

from pydantic import BaseModel, Field

from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import AnyMessage, AssistantMessage, UserMessage
from beeai_framework.workflows.v2.decorators.after import after
from beeai_framework.workflows.v2.decorators.end import end
from beeai_framework.workflows.v2.decorators.start import start
from beeai_framework.workflows.v2.workflow import Workflow


def answer_prompt(user_message: AnyMessage, thoughts: str) -> str:
    return f""""You have access to the internal reasoning about the user's message.
Generate a clear, concise, and contextually appropriate reply based on that reasoning.
Do not introduce unrelated ideas; your answer should directly reflect the thought process that was internally generated.
Internal reasoning: {thoughts}
User's Message: {user_message.text}"""


class ResponseWithConfidence(BaseModel):
    response: str
    confidence: int = Field(ge=1, le=10, description="Confidence in the correctness of your response between 1 and 10.")


class RespondWithConfidenceWorkflow(Workflow):
    def __init__(self) -> None:
        super().__init__()
        self.chat_model: ChatModel = ChatModel.from_name("ollama:granite3.3:8b")
        self.retries: int = 3

    @start
    async def start(self, input: list[AnyMessage]) -> list[AnyMessage]:
        print("Start")
        return input

    @after(start)
    async def answer(self, messages: list[AnyMessage]) -> ResponseWithConfidence:
        print("Generating response")
        output = await self.chat_model.run(messages, response_format=ResponseWithConfidence)
        assert output.output_structured is not None
        response = ResponseWithConfidence(**output.output_structured.model_dump())
        return response

    @after(answer)
    @end
    async def end(self, response: ResponseWithConfidence) -> list[AnyMessage]:
        content = f"{response.response}\nConfidence: {response.confidence}/10"
        return [AssistantMessage(content)]


# Async main function
async def main() -> None:
    workflow = RespondWithConfidenceWorkflow()
    output = await workflow.run([UserMessage("What is at the center of a black hole?")])
    print(output.last_message.text)


# Entry point
if __name__ == "__main__":
    asyncio.run(main())
