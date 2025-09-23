# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio

from pydantic import BaseModel

from beeai_framework.backend.message import AnyMessage, UserMessage
from beeai_framework.workflows.v2.decorators.end import end
from beeai_framework.workflows.v2.decorators.start import start
from beeai_framework.workflows.v2.workflow import Workflow


class Page(BaseModel):
    link: str
    content: str


class WebScrapperWorkflow(Workflow):
    def __init__(self) -> None:
        super().__init__()

    @start
    @end
    async def identity(self, messages: list[AnyMessage]) -> list[AnyMessage]:
        return []


# Async main function
async def main() -> None:
    flow = WebScrapperWorkflow()
    messages = await flow.run(
        [
            UserMessage(
                "Imagine we receive a signal from an intelligent extraterrestrial civilization. How should we interpret it, what assumptions should we question, and what could be the global implications of responding?"
            )
        ]
    )

    for m in messages:
        print(f"{m.role}: {m.text}")


# Entry point
if __name__ == "__main__":
    asyncio.run(main())
