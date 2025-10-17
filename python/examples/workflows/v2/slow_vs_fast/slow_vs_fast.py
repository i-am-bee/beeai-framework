# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio
from pathlib import Path
from typing import Any

from beeai_framework.backend.message import AnyMessage, AssistantMessage, UserMessage
from beeai_framework.workflows.v2.decorators._and import _and
from beeai_framework.workflows.v2.decorators._or import _or
from beeai_framework.workflows.v2.decorators.after import after
from beeai_framework.workflows.v2.decorators.end import end
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

    @after(_or(start, "fast"))
    @when(lambda self, messages, _: self.fast_runs > 0)
    async def fast(self, messages: list[AnyMessage], _: Any) -> None:
        """Fast running operation"""
        await asyncio.sleep(1)
        self.fast_runs -= 1
        print("Fast complete!")

    @after(_and(slow, fast))
    @end
    async def end(self, slow: Any, fast: Any) -> list[AnyMessage]:
        return [AssistantMessage("Fast and slow complete!")]


# Async main function
async def main() -> None:
    workflow = SlowVsFastWorkflow()
    workflow.print_html(Path(__file__).resolve().parent / "workflow.html")
    output = await workflow.run([UserMessage("Hello")])
    print(output.last_message.text)


# Entry point
if __name__ == "__main__":
    asyncio.run(main())
