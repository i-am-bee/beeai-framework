# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio

from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import AnyMessage, AssistantMessage, UserMessage
from beeai_framework.backend.types import ChatModelOutput
from beeai_framework.workflows.v2.decorators.after import after
from beeai_framework.workflows.v2.decorators.start import start
from beeai_framework.workflows.v2.workflow import Workflow


def thinking_prompt(user_message: AnyMessage) -> str:
    return f""""Given a user message, analyze and reason about it deeply.
Do not generate a reply. Focus entirely on understanding implications, context, assumptions, and possible interpretations.
User's Message: {user_message.text}"""


def answer_prompt(user_message: AnyMessage, thoughts: str) -> str:
    return f""""You have access to the internal reasoning about the user's message.
Generate a clear, concise, and contextually appropriate reply based on that reasoning.
Do not introduce unrelated ideas; your answer should directly reflect the thought process that was internally generated.
Internal reasoning: {thoughts}
User's Message: {user_message.text}"""


class ThinkBeforeAnswerWorkflow(Workflow):
    def __init__(self) -> None:
        super().__init__()
        self.chat_model: ChatModel = ChatModel.from_name("ollama:granite3.3:8b")

    @start
    async def start(self, input: list[AnyMessage]) -> AnyMessage:
        print("Start")
        return input[-1]

    @after(start)
    async def think(self, user_message: AnyMessage) -> str:
        print("Thinking")
        prompt = thinking_prompt(user_message)
        output: ChatModelOutput = await self.chat_model.run([UserMessage(content=prompt)])
        return output.get_text_content()

    @after(start, think)
    async def answer(self, user_message: AnyMessage, thoughts: str) -> AssistantMessage:
        print("Answering")
        prompt = answer_prompt(user_message, thoughts)
        output: ChatModelOutput = await self.chat_model.run([UserMessage(content=prompt)])
        return AssistantMessage(output.get_text_content())

    @after(answer)
    async def end(self, msg: AssistantMessage) -> None:
        print("End")
        self._messages.append(msg)


# Async main function
async def main() -> None:
    flow = ThinkBeforeAnswerWorkflow()
    messages = await flow.run(
        [
            UserMessage(
                "Imagine we receive a signal from an intelligent extraterrestrial civilization. How should we interpret it, what assumptions should we question, and what could be the global implications of responding?"
            )
        ]
    )
    print(messages[-1].text)


# Entry point
if __name__ == "__main__":
    asyncio.run(main())
