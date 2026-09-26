# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio
from contextvars import Context
from io import StringIO
from threading import Event

import pytest
from typing_extensions import Unpack

from beeai_framework.utils.io import IOConfirmKwargs, io_confirm, io_read, setup_io_context


@pytest.mark.unit
def test_read_default_in_new_context(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stdin", StringIO("hello\n"))
    assert Context().run(asyncio.run, io_read("Read: ")) == "hello"


@pytest.mark.unit
@pytest.mark.parametrize(("answer", "expected"), [("YES\n", True), ("no\n", False)])
def test_confirm_default_in_new_context(monkeypatch: pytest.MonkeyPatch, answer: str, expected: bool) -> None:
    monkeypatch.setattr("sys.stdin", StringIO(answer))
    assert Context().run(asyncio.run, io_confirm("Confirm: ", title="Permission")) is expected


@pytest.mark.unit
def test_confirm_default_does_not_block_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    input_ready = Event()

    class PendingInput(StringIO):
        def readline(self, size: int = -1) -> str:
            if not input_ready.wait(timeout=5):
                raise TimeoutError("The event loop could not make input available while waiting for confirmation")
            return super().readline(size)

    async def confirm() -> bool:
        asyncio.get_running_loop().call_soon(input_ready.set)
        return await io_confirm("Confirm: ", title="Permission")

    monkeypatch.setattr("sys.stdin", PendingInput("yes\n"))
    try:
        assert Context().run(asyncio.run, confirm()) is True
    finally:
        input_ready.set()


@pytest.mark.unit
def test_handler_cleanup_in_new_context(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stdin", StringIO("other context\nrestored\nyes\n"))
    context = Context()

    async def read(prompt: str) -> str:
        return f"override: {prompt}"

    async def nested_read(prompt: str) -> str:
        return f"nested: {prompt}"

    async def confirm(prompt: str, **kwargs: Unpack[IOConfirmKwargs]) -> bool:
        return kwargs.get("title") == "allowed"

    # pyrefly: ignore [bad-argument-type]
    cleanup = context.run(setup_io_context, read=read, confirm=confirm)
    try:
        assert context.run(asyncio.run, io_read("question")) == "override: question"
        assert context.run(asyncio.run, io_confirm("question", title="allowed")) is True
        assert context.run(asyncio.run, io_confirm("question", title="denied")) is False
        assert Context().run(asyncio.run, io_read("")) == "other context"
        # pyrefly: ignore [bad-argument-type]
        nested_cleanup = context.run(setup_io_context, read=nested_read, confirm=confirm)
        try:
            assert context.run(asyncio.run, io_read("question")) == "nested: question"
        finally:
            context.run(nested_cleanup)
        assert context.run(asyncio.run, io_read("question")) == "override: question"
    finally:
        context.run(cleanup)

    assert context.run(asyncio.run, io_read("")) == "restored"
    assert context.run(asyncio.run, io_confirm("")) is True
