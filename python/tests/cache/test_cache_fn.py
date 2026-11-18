# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest

from beeai_framework.cache import CacheFn, decorator_cache


@pytest.mark.unit
@pytest.mark.asyncio
async def test_slow_result_is_cached_for_its_full_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = Mock()
    now = 0.0
    clock.time.side_effect = lambda: now
    monkeypatch.setattr(decorator_cache, "time", clock)
    calls = 0

    async def slow_tool() -> int:
        nonlocal calls, now
        calls += 1
        now += 20
        return calls

    cached_tool = CacheFn.create(slow_tool, default_ttl=10)
    assert await cached_tool() == 1
    now += 9
    assert await cached_tool() == 1
    now += 1
    assert await cached_tool() == 2
