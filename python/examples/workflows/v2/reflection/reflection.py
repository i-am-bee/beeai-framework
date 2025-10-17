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


class Reflection(BaseModel):
    response: str
    critique: str


class Critique(BaseModel):
    critique: str = Field(description="A helpful critique of the most recent assistant message.")


def sys_prompt(
    reflection: Reflection | None = None,
) -> str:
    prompt = "You are a helpful and knowledgeable AI assistant that provides accurate, clear, and concise responses to user queries."

    if reflection:
        prompt += f"""
Here is your previous response and a helpful critique.
Your new response should be an iterative improvement of your previous response, taking the critique into account.

Previous Response: {reflection.response}
Critique: {reflection.critique}
"""
    return prompt


def critique_sys_prompt() -> str:
    return """You are a helpful critic. Analyze the last assistant message, assess its quality, and provide a constructive critique."""


class SelfReflectionWorkflow(Workflow):
    def __init__(self) -> None:
        super().__init__()
        self.num_iterations = 3
        self.chat_model: ChatModel = ChatModel.from_name("ollama:ibm/granite4")
        self.critique_model: ChatModel = ChatModel.from_name("ollama:gpt-oss:20b")

    @start
    async def start(self, input: list[AnyMessage]) -> list[AnyMessage]:
        return input

    @after(start)
    async def no_reflection_answer(self, messages: list[AnyMessage]) -> None:
        """Print the default response for comparison"""
        output = await self.chat_model.run([SystemMessage(content=sys_prompt()), *messages])
        print(f"{output.get_text_content()}\n==========")

    @after(_or(start, "reflect"))
    async def answer(self, messages: list[AnyMessage], reflection: Reflection) -> str:
        """Generate response"""
        output = await self.chat_model.run([SystemMessage(content=sys_prompt(reflection=reflection)), *messages])
        return output.get_text_content()

    @after(answer)
    @when(lambda self, response: self.num_iterations > 0)
    async def reflect(self, response: str) -> Reflection:
        """Reflect on the response"""
        self.num_iterations -= 1
        last_exec = self.inspect(self.start).last_execution()
        raw_inputs = last_exec.inputs[0] if last_exec is not None else []
        messages: list[AnyMessage] = cast(list[AnyMessage], raw_inputs)

        output = await self.critique_model.run(
            [SystemMessage(content=critique_sys_prompt()), *messages, AssistantMessage(content=response)],
            response_format=Critique,
        )
        assert output.output_structured is not None
        critique = Critique(**output.output_structured.model_dump())
        print(f"Critique -> {critique.critique}")
        return Reflection(response=response, critique=critique.critique)

    @end
    @after(answer)
    @when(lambda self, response: self.num_iterations <= 0)
    async def end(self, response: str) -> list[AnyMessage]:
        return [AssistantMessage(response)]


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
    print(f"{output.last_message.text}")


# Entry point
if __name__ == "__main__":
    asyncio.run(main())
