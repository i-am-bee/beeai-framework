# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio
from pathlib import Path

from pydantic import BaseModel, Field

from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import AnyMessage, AssistantMessage, UserMessage
from beeai_framework.workflows.v2.decorators.after import after
from beeai_framework.workflows.v2.decorators.end import end
from beeai_framework.workflows.v2.decorators.start import start
from beeai_framework.workflows.v2.workflow import Workflow


class ResponseWithConfidence(BaseModel):
    response: str = Field(description="Comprehensive response.")
    confidence: int = Field(
        description="How confident are you in the correctness of the response? Chose a value between 1 and 10, 1 being lowest, 10 being highest."
    )


class RespondWithConfidenceWorkflow(Workflow):
    def __init__(self) -> None:
        super().__init__()
        self.chat_model: ChatModel = ChatModel.from_name("ollama:gpt-oss:20b")

    @start
    async def start(self, input: list[AnyMessage]) -> list[AnyMessage]:
        print("Start")
        return input

    @after(start)
    async def answer(self, messages: list[AnyMessage]) -> ResponseWithConfidence:
        print("Generating response")
        output = await self.chat_model.run(messages, response_format=ResponseWithConfidence)
        assert output.output_structured is not None
        return ResponseWithConfidence(**output.output_structured.model_dump())

    @after(answer)
    @end
    async def end(self, response: ResponseWithConfidence) -> list[AnyMessage]:
        content = f"{response.response}\nConfidence: {response.confidence}/10"
        return [AssistantMessage(content)]


# Async main function
async def main() -> None:
    workflow = RespondWithConfidenceWorkflow()
    workflow.print_html(Path(__file__).resolve().parent / "workflow.html")
    output = await workflow.run([UserMessage("What is at the center of a black hole?")])
    print(output.last_message.text)
    output = await workflow.run([UserMessage("What is 10 + 10?")])
    print(output.last_message.text)


# Entry point
if __name__ == "__main__":
    asyncio.run(main())
