# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio
import random
from pathlib import Path
from typing import Any

from beeai_framework.backend.message import AnyMessage, UserMessage
from beeai_framework.context import RunMiddlewareType
from beeai_framework.workflows.v2.decorators.after import after
from beeai_framework.workflows.v2.decorators.end import end
from beeai_framework.workflows.v2.decorators.retry import retry
from beeai_framework.workflows.v2.decorators.start import start
from beeai_framework.workflows.v2.workflow import Workflow


class RetryWorkflow(Workflow):
    def __init__(self, middlewares: list[RunMiddlewareType] | None = None) -> None:
        super().__init__(middlewares=middlewares)

    @start
    async def start(self, input: list[AnyMessage]) -> list[AnyMessage]:
        return input

    @after(start)
    @retry(3)
    async def do_something_flaky(self, messages: list[AnyMessage]) -> None:
        print("do_something_flaky")
        if random.random() < 0.5:
            raise ValueError("Random failure!")

    @after(do_something_flaky)
    @end
    async def end(self, flaky_output: Any) -> list[AnyMessage]:
        return []


async def main() -> None:
    workflow = RetryWorkflow()
    workflow.print_html(Path(__file__).resolve().parent / "workflow.html")
    output = await workflow.run([UserMessage("Hello")])
    print(output.last_message.text)


# Entry point
if __name__ == "__main__":
    asyncio.run(main())
