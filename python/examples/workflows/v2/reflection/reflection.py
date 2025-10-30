# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio
from pathlib import Path
from typing import cast

from pydantic import BaseModel, Field

from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import AnyMessage, AssistantMessage, SystemMessage, UserMessage
from beeai_framework.workflows.v2.decorators._or import _or
from beeai_framework.workflows.v2.decorators.after import after
from beeai_framework.workflows.v2.decorators.end import end
from beeai_framework.workflows.v2.decorators.start import start
from beeai_framework.workflows.v2.decorators.when import when
from beeai_framework.workflows.v2.workflow import Workflow


class ResponseWithReflection(BaseModel):
    response: str
    reflection: str = Field(description="A helpful critique of the most recent assistant message.")


def sys_prompt(
    reflection: ResponseWithReflection | None = None,
) -> str:
    prompt = "You are a helpful and knowledgeable AI assistant that provides accurate, clear, and concise responses to user queries."

    if reflection:
        prompt += f"""
Here is your previous response and a helpful critique.
Your response should be an iterative improvement of your previous response, taking the critique into account.

Previous Response: {reflection.response}
Critique: {reflection.reflection}
"""
    return prompt


def reflect_prompt() -> str:
    return """Analyze the last assistant response, assess its quality, limit your review to 2 lines including suggestions for improvement."""


class SelfReflectionWorkflow(Workflow):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[AnyMessage] = []
        self.num_iterations = 3
        self.chat_model: ChatModel = ChatModel.from_name("ollama:ibm/granite4")
        self.reflect_model: ChatModel = ChatModel.from_name("ollama:ibm/granite4")

    @start
    async def start(self, input: list[AnyMessage]) -> None:
        print("Start")
        self.messages = input
        self.response: str | None = None

    @after(start)
    async def answer(self) -> str:
        """Generate response"""
        output = await self.chat_model.run([SystemMessage(content=sys_prompt()), *self.messages])
        self.response = output.get_text_content()
        print("\nAnswer", ("*" * 20), "\n")
        print(self.response)
        return self.response

    @after("reflect")
    async def answer_with_reflection(self, reflection: ResponseWithReflection) -> str:
        """Generate response"""
        output = await self.chat_model.run([SystemMessage(content=sys_prompt(reflection=reflection)), *self.messages])
        self.response = output.get_text_content()
        print("\nAnswer + reflection", ("*" * 20), "\n")
        print(self.response)
        return self.response

    @after(_or(answer, answer_with_reflection))
    @when(lambda self: self.num_iterations > 0)
    async def reflect(self, response: str) -> ResponseWithReflection:
        """Reflect on the response"""
        self.num_iterations -= 1
        last_exec = self.inspect(self.start).last_execution()
        raw_inputs = last_exec.inputs[0] if last_exec is not None else []
        messages: list[AnyMessage] = cast(list[AnyMessage], raw_inputs)

        output = await self.reflect_model.run(
            [*messages, AssistantMessage(content=response), UserMessage(content=reflect_prompt())],
        )
        print("\nReflection", ("*" * 20), "\n")
        print(output.get_text_content())
        return ResponseWithReflection(response=response, reflection=output.get_text_content())

    @end
    @after(_or(answer, answer_with_reflection))
    @when(lambda self: self.num_iterations <= 0)
    async def end(self) -> list[AnyMessage]:
        return [AssistantMessage(self.response or "")]


# Async main function
async def main() -> None:
    workflow = SelfReflectionWorkflow()
    workflow.print_html(Path(__file__).resolve().parent / "workflow.html")
    output = await workflow.run(
        [
            UserMessage(
                content="If human memory is reconstructive rather than reproductive, how might that influence the reliability of eyewitness testimony in court?"
            )
        ]
    )
    print("\nFinal answer", ("*" * 20), "\n")
    print(f"{output.last_message.text}")


# Entry point
if __name__ == "__main__":
    asyncio.run(main())
