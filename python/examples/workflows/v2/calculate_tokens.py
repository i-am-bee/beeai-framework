# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio
import re

from beeai_framework.backend.message import AnyMessage, AssistantMessage, UserMessage
from beeai_framework.workflows.v2.decorators.after import after
from beeai_framework.workflows.v2.decorators.start import start
from beeai_framework.workflows.v2.decorators.when import when
from beeai_framework.workflows.v2.workflow import Workflow


class CalculateTokensWorkflow(Workflow):
    def __init__(self) -> None:
        super().__init__()

    @start
    async def convert_to_text(self, messages: list[AnyMessage]) -> str:
        return "".join(msg.text for msg in messages)

    @after(convert_to_text)
    @when(lambda _, text: len(text) < 1000)
    async def count_tokens_by_whitespaces(self, text: str) -> int:
        print("count_tokens_by_whitespaces")
        return len(text.split(" "))

    @after(convert_to_text)
    @when(lambda _, text: len(text) >= 1000)
    async def count_tokens_regex(self, text: str) -> int:
        print("count_tokens_regex")
        tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        return len(tokens)

    @after(count_tokens_by_whitespaces, count_tokens_regex, type="OR")
    async def finalize(self, white_space_tokens: int | None = None, count_regex_tokens: int | None = None) -> None:
        token_count = 0

        if white_space_tokens is not None:
            token_count = white_space_tokens
        elif count_regex_tokens is not None:
            token_count = count_regex_tokens

        self._messages.append(AssistantMessage(f"Total tokens: {token_count}"))


# Async main function
async def main() -> None:
    flow = CalculateTokensWorkflow()
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
