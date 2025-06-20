# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0


import pytest
import pytest_asyncio

from beeai_framework.cache import UnconstrainedCache


@pytest_asyncio.fixture
async def cache() -> UnconstrainedCache[str]:
    _cache: UnconstrainedCache[str] = UnconstrainedCache()
    await _cache.set("key1", "value1")
    await _cache.set("key2", "value2")
    await _cache.set("key3", "value3")
    return _cache


@pytest.mark.asyncio
@pytest.mark.unit
async def test_cache_size(cache: UnconstrainedCache[str]) -> None:
    assert cache.enabled
    assert await cache.size() == 3


@pytest.mark.asyncio
@pytest.mark.unit
async def test_cache_set(cache: UnconstrainedCache[str]) -> None:
    await cache.set("key4", "value4")
    await cache.set("key5", "value5")

    assert await cache.size() == 5


@pytest.mark.asyncio
@pytest.mark.unit
async def test_cache_get(cache: UnconstrainedCache[str]) -> None:
    value0 = await cache.get("key0")
    value2 = await cache.get("key2")

    assert value0 is None
    assert value2 == "value2"

    assert await cache.size() == 3


@pytest.mark.asyncio
@pytest.mark.unit
async def test_cache_has(cache: UnconstrainedCache[str]) -> None:
    assert await cache.has("key1")
    assert await cache.has("key4") is False


@pytest.mark.asyncio
@pytest.mark.unit
async def test_cache_delete(cache: UnconstrainedCache[str]) -> None:
    del0 = await cache.delete("key0")
    del2 = await cache.delete("key2")

    assert del0 is False
    assert del2 is True
    assert await cache.size() == 2


@pytest.mark.asyncio
@pytest.mark.unit
async def test_cache_clear(cache: UnconstrainedCache[str]) -> None:
    assert await cache.size() == 3
    await cache.clear()
    assert await cache.size() == 0
