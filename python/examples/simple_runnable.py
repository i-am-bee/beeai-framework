import asyncio
from functools import cached_property
from typing import Unpack

from beeai_framework.backend import AnyMessage, AssistantMessage, UserMessage
from beeai_framework.context import RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.runnable import Runnable, RunnableOptions, RunnableOutput, runnable_entry


class GreetingRunnable(Runnable[RunnableOutput]):
    @runnable_entry
    async def run(self, input: list[AnyMessage], /, **kwargs: Unpack[RunnableOptions]) -> RunnableOutput:
        # retrieves the current run contex
        run = RunContext.get()

        response = f"Hello, {run.context.get('name', 'stranger')}!"

        # sends an emit so that someone can react to it (optional)
        await run.emitter.emit("before_send", response)

        return RunnableOutput(output=[AssistantMessage(response)])

    @cached_property
    def emitter(self) -> Emitter:
        return Emitter.root().child(namespace=["echo"])


async def main() -> None:
    echo = GreetingRunnable()
    response = await echo.run([UserMessage("Hello!")]).context({"name": "Alex"})
    print(response.last_message.text)


if __name__ == "__main__":
    asyncio.run(main())
