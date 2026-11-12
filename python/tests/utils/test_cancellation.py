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


@pytest.mark.unit
def test_abort_cleanup_does_not_skip_other_children() -> None:
    source = AbortController()
    children = [AbortController() for _ in range(3)]
    cleanups = [register_signals(child, [source.signal]) for child in children]
    for child, cleanup in zip(children, cleanups, strict=True):
        child.signal.add_event_listener(cleanup)

    source.abort("stop all")

    assert all(child.signal.aborted for child in children)
    assert all(child.signal.reason == "stop all" for child in children)
    assert not source.signal._listeners


@pytest.mark.unit
def test_self_removing_listener_does_not_skip_next_listener() -> None:
    controller = AbortController()
    calls: list[str] = []

    def first() -> None:
        calls.append("first")
        controller.signal.remove_event_listener(first)

    controller.signal.add_event_listener(first)
    controller.signal.add_event_listener(lambda: calls.append("second"))

    controller.abort()

    assert calls == ["first", "second"]


@pytest.mark.unit
def test_listener_removed_before_its_turn_is_not_called() -> None:
    controller = AbortController()
    calls: list[str] = []

    def removed() -> None:
        calls.append("removed")

    controller.signal.add_event_listener(lambda: controller.signal.remove_event_listener(removed))
    controller.signal.add_event_listener(removed)
    controller.signal.add_event_listener(lambda: calls.append("last"))

    controller.abort()

    assert calls == ["last"]


@pytest.mark.unit
def test_new_listener_waits_until_the_next_abort() -> None:
    controller = AbortController()
    calls: list[str] = []

    def added() -> None:
        calls.append("added")

    def first() -> None:
        calls.append("first")
        controller.signal.remove_event_listener(first)
        controller.signal.add_event_listener(added)

    controller.signal.add_event_listener(first)
    controller.signal.add_event_listener(lambda: calls.append("last"))

    controller.abort("first reason")
    assert calls == ["first", "last"]

    controller.abort("second reason")
    assert calls == ["first", "last", "last", "added"]
    assert controller.signal.reason == "second reason"


@pytest.mark.unit
def test_abort_listener_error_keeps_existing_propagation() -> None:
    controller = AbortController()
    calls: list[str] = []
    error = ValueError("listener failed")

    def failing() -> None:
        raise error

    controller.signal.add_event_listener(failing)
    controller.signal.add_event_listener(lambda: calls.append("later"))

    with pytest.raises(ValueError) as exc:
        controller.abort("stop")

    assert exc.value is error
    assert calls == []
    assert controller.signal.aborted
    assert controller.signal.reason == "stop"


@pytest.mark.unit
def test_duplicate_callback_registrations_are_independent() -> None:
    controller = AbortController()
    calls: list[str] = []

    def duplicate() -> None:
        calls.append("duplicate")

    controller.signal.add_event_listener(duplicate)
    controller.signal.add_event_listener(duplicate)
    controller.abort()
    assert calls == ["duplicate", "duplicate"]

    controller.signal.remove_event_listener(duplicate)
    controller.abort()
    assert calls == ["duplicate", "duplicate", "duplicate"]


@pytest.mark.unit
def test_removing_one_duplicate_during_abort_keeps_one_registration() -> None:
    controller = AbortController()
    calls: list[str] = []

    def duplicate() -> None:
        calls.append("duplicate")

    controller.signal.add_event_listener(lambda: controller.signal.remove_event_listener(duplicate))
    controller.signal.add_event_listener(duplicate)
    controller.signal.add_event_listener(duplicate)

    controller.abort()

    assert calls == ["duplicate"]


@pytest.mark.unit
def test_removed_and_reregistered_callback_waits_until_next_abort() -> None:
    controller = AbortController()
    calls: list[str] = []

    def moved() -> None:
        calls.append("moved")

    def first() -> None:
        calls.append("first")
        controller.signal.remove_event_listener(first)
        controller.signal.remove_event_listener(moved)
        controller.signal.add_event_listener(moved)

    controller.signal.add_event_listener(first)
    controller.signal.add_event_listener(moved)
    controller.signal.add_event_listener(lambda: calls.append("last"))

    controller.abort()
    assert calls == ["first", "last"]

    controller.abort()
    assert calls == ["first", "last", "last", "moved"]


@pytest.mark.unit
def test_remove_listener_preserves_identity_matching() -> None:
    controller = AbortController()
    calls: list[str] = []

    class Callback:
        def __call__(self) -> None:
            calls.append("called")

        def __eq__(self, other: object) -> bool:
            return False

    callback = Callback()
    controller.signal.add_event_listener(callback)
    controller.signal.remove_event_listener(callback)
    controller.abort()

    assert calls == []
