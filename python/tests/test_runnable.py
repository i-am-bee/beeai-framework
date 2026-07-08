# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Unpack

import pytest

from beeai_framework.backend import AnyMessage, AssistantMessage, UserMessage
from beeai_framework.context import RunContext, RunMiddlewareType
from beeai_framework.emitter import Emitter, EventMeta
from beeai_framework.runnable import Runnable, RunnableOptions, RunnableOutput, runnable_entry

"""
Utility functions and classes
"""


class MyRunnable(Runnable[RunnableOutput]):
    def __init__(self, middlewares: list[RunMiddlewareType] | None = None) -> None:
        super().__init__(middlewares)

    @property
    def emitter(self) -> Emitter:
        return Emitter.root().child(namespace=["runnable", "main"])

    @runnable_entry
    async def run(self, input: list[AnyMessage], /, **kwargs: Unpack[RunnableOptions]) -> RunnableOutput:
        ctx = RunContext.get()
        await ctx.emitter.emit("run", f"input: {input[-1].text}, context: {kwargs.get('context')}")
        return RunnableOutput(output=[AssistantMessage(content="Hi, there!")])


async def process_runnable_events(event_data: Any, event_meta: EventMeta) -> None:
    print(event_data, event_meta)


async def observer(emitter: Emitter) -> None:
    emitter.on("*.*", process_runnable_events)


"""
Unit Tests
"""


@pytest.mark.asyncio
@pytest.mark.unit
async def test_runnable() -> None:
    r = MyRunnable()
    o = await r.run([UserMessage(content="Hi!")]).observe(observer)
    assert o.output[-1].text == "Hi, there!"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_runnable_with_kwargs() -> None:
    r = MyRunnable()
    o = await r.run([UserMessage(content="Hi!")], context={"tags": ["test"]})
    assert o.output[-1].text == "Hi, there!"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_runnable_with_observer() -> None:
    r = MyRunnable()
    o = await r.run([UserMessage(content="Hi!")], context={"tags": ["test"]}).observe(observer)
    assert o.output[-1].text == "Hi, there!"
