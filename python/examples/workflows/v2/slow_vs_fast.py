# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio
from typing import Any

from beeai_framework.backend.message import AnyMessage, UserMessage
from beeai_framework.workflows.v2.decorators.after import after
from beeai_framework.workflows.v2.decorators.start import start
from beeai_framework.workflows.v2.decorators.when import when
from beeai_framework.workflows.v2.workflow import Workflow


class SlowVsFastWorkflow(Workflow):
    def __init__(self) -> None:
        super().__init__()
        self.fast_runs = 5

    @start
    async def start(self, input: list[AnyMessage]) -> list[AnyMessage]:
        return input

    @after(start)
    async def slow(self, messages: list[AnyMessage]) -> None:
        """Slow running operation"""
        await asyncio.sleep(10)
        print("Slow complete!")

    @after(start, "fast", type="OR")
    @when(lambda self, messages, _: self.fast_runs > 0)
    async def fast(self, messages: list[AnyMessage], _: Any) -> None:
        """Fast running operation"""
        await asyncio.sleep(1)
        self.fast_runs -= 1
        print("Fast complete!")

    @after(slow, fast)
    async def end(self, slow: Any, fast: Any) -> None:
        print("Fast and slow complete!")


# Async main function
async def main() -> None:
    flow = SlowVsFastWorkflow()
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
