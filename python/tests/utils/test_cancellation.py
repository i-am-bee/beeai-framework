# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from beeai_framework.utils import AbortSignal


@pytest.mark.unit
@pytest.mark.asyncio
async def test_timeout_reason_uses_seconds() -> None:
    # loop.call_later() takes seconds, so the abort reason must not label the
    # duration as milliseconds.
    signal = AbortSignal.timeout(0.01)
    await asyncio.sleep(0.05)

    assert signal.aborted
    assert signal.reason == "Operation timed out after 0.01 s"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_manual_abort_reason() -> None:
    signal = AbortSignal()
    signal._abort("custom reason")

    assert signal.aborted
    assert signal.reason == "custom reason"
