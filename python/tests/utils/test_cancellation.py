# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import pytest

from beeai_framework.utils.cancellation import AbortController


@pytest.mark.unit
@pytest.mark.asyncio
async def test_clone_carries_the_abort_state() -> None:
    controller = AbortController()
    controller.abort("boom")

    cloned = await controller.clone()

    assert cloned.signal.aborted is True
    assert cloned.signal.reason == "boom"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_clone_is_independent() -> None:
    controller = AbortController()
    cloned = await controller.clone()
    assert cloned.signal.aborted is False

    fired: list[str] = []
    controller.signal.add_event_listener(lambda: fired.append("original"))

    cloned.abort("only the clone")

    assert controller.signal.aborted is False
    assert fired == []
