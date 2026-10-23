# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import pytest

from beeai_framework.utils.cancellation import AbortController, register_signals


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


@pytest.mark.unit
def test_signal_cleanup_preserves_other_subscribers() -> None:
    source = AbortController()
    completed = AbortController()
    active = AbortController()
    notifications: list[str] = []
    source.signal.add_event_listener(lambda: notifications.append("aborted"))
    cleanup_completed = register_signals(completed, [source.signal])
    cleanup_active = register_signals(active, [source.signal])

    cleanup_completed()
    cleanup_completed()
    source.abort("stop")

    assert not completed.signal.aborted
    assert active.signal.aborted
    assert active.signal.reason == "stop"
    assert notifications == ["aborted"]
    cleanup_active()
    assert len(source.signal._listeners) == 1


@pytest.mark.unit
def test_signal_cleanup_detaches_from_all_sources() -> None:
    sources = [AbortController(), AbortController()]
    child = AbortController()
    cleanup = register_signals(child, [source.signal for source in sources])

    cleanup()
    for source in sources:
        assert not source.signal._listeners
        source.abort("stop")
    assert not child.signal.aborted


@pytest.mark.unit
def test_register_signals_preserves_pre_aborted_source() -> None:
    source = AbortController()
    source.abort("already stopped")
    child = AbortController()

    cleanup = register_signals(child, [source.signal])

    assert child.signal.aborted
    assert child.signal.reason == "already stopped"
    assert not source.signal._listeners
    cleanup()
    cleanup()
