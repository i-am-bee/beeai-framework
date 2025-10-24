# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio
import random
from pathlib import Path
from typing import Any

from beeai_framework.backend.message import AnyMessage, AssistantMessage, UserMessage
from beeai_framework.workflows.v2.decorators._or import _or
from beeai_framework.workflows.v2.decorators.after import after
from beeai_framework.workflows.v2.decorators.end import end
from beeai_framework.workflows.v2.decorators.start import start
from beeai_framework.workflows.v2.decorators.when import when
from beeai_framework.workflows.v2.workflow import Workflow


class LoopUntilDoneWorkflow(Workflow):
    def __init__(self) -> None:
        super().__init__()
        self.complete = False

    @start
    async def start(self, input: list[AnyMessage]) -> list[AnyMessage]:
        print("Start")
        return input

    @after(_or(start, "loop"))
    @when(lambda self, messages, _: not self.complete)
    async def loop(self, messages: list[AnyMessage], _: Any) -> None:
        num = random.random()
        print(num)
        if num < 0.1:
            self.complete = True

    @end
    async def end(self) -> list[AnyMessage]:
        print("Done!")
        return [AssistantMessage("Done!")]


# Async main function
async def main() -> None:
    workflow = LoopUntilDoneWorkflow()
    workflow.print_html(Path(__file__).resolve().parent / "workflow.html")
    output = await workflow.run([UserMessage("Hello!")])
    print(output.last_message.text)


# Entry point
if __name__ == "__main__":
    asyncio.run(main())
