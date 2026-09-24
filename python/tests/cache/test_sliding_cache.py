# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio
from threading import Lock

import pytest
import pytest_asyncio
from cachetools import TTLCache

from beeai_framework.cache import SlidingCache


@pytest_asyncio.fixture
async def sized_cache() -> SlidingCache[str]:
    _cache: SlidingCache[str] = SlidingCache(size=4)
    await _cache.set("key1", "value1")
    await _cache.set("key2", "value2")
    await _cache.set("key3", "value3")
    return _cache


@pytest_asyncio.fixture
async def timed_cache() -> SlidingCache[str]:
    _cache: SlidingCache[str] = SlidingCache(size=4, ttl=3)
    await _cache.set("key1", "value1")
    await _cache.set("key2", "value2")
    await _cache.set("key3", "value3")
    return _cache


@pytest.mark.asyncio
@pytest.mark.unit
async def test_cache_size(sized_cache: SlidingCache[str]) -> None:
    assert sized_cache.enabled
    assert await sized_cache.size() == 3

    await sized_cache.set("key4", "value4")
    await sized_cache.set("key5", "value5")

    assert await sized_cache.size() == 4


@pytest.mark.asyncio
@pytest.mark.unit
async def test_cache_get(sized_cache: SlidingCache[str]) -> None:
    value5 = await sized_cache.get("key5")
    value2 = await sized_cache.get("key2")

    assert value5 is None
    assert value2 == "value2"

    assert await sized_cache.size() == 3


@pytest.mark.asyncio
@pytest.mark.unit
async def test_cache_has(sized_cache: SlidingCache[str]) -> None:
    assert await sized_cache.has("key1")
    assert await sized_cache.has("key4") is False


@pytest.mark.asyncio
@pytest.mark.unit
async def test_cache_delete(sized_cache: SlidingCache[str]) -> None:
    del0 = await sized_cache.delete("key0")
    del2 = await sized_cache.delete("key2")

    assert del0 is False
    assert del2 is True
    assert await sized_cache.size() == 2

    await sized_cache.set("key4", "value4")
    await sized_cache.set("key5", "value5")
    await sized_cache.set("key6", "value6")

    assert await sized_cache.size() == 4


@pytest.mark.asyncio
@pytest.mark.unit
async def test_cache_clear(sized_cache: SlidingCache[str]) -> None:
    assert await sized_cache.size() == 3
    await sized_cache.clear()
    assert await sized_cache.size() == 0


@pytest.mark.asyncio
@pytest.mark.unit
async def test_cache_timed(timed_cache: SlidingCache[str]) -> None:
    assert await timed_cache.size() == 3
    await asyncio.sleep(3)
    assert await timed_cache.size() == 0


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("ttl", [None, 60])
async def test_cache_clone_independent(ttl: float | None) -> None:
    cache: SlidingCache[dict[str, int]] = SlidingCache(size=2, ttl=ttl)
    value = {"value": 1}
    await cache.set("key1", value)
    cloned = await cache.clone()

    assert await cloned.get("key1") is value
    await cloned.set("key1", {"value": 2})
    assert await cache.get("key1") is value
    await cache.set("key2", value)
    assert not await cloned.has("key2")
    assert await cloned.delete("key1")
    assert await cache.has("key1")
    await cloned.set("key3", value)
    await cloned.clear()
    assert await cache.size() == 2
    await cache.clear()
    for key in ("key4", "key5"):
        await cache.set(key, value)
        await cloned.set(key, value)
    assert await cache.size() == await cloned.size() == 2


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("ttl", [None, 60])
async def test_cache_clone_empty(ttl: float | None) -> None:
    cache: SlidingCache[str] = SlidingCache(size=2, ttl=ttl)
    cloned = await cache.clone()
    await cloned.set("key1", "value1")
    await cloned.set("key2", "value2")
    assert await cloned.size() == 2
    assert await cache.size() == 0


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("ttl", [None, 60])
async def test_cache_clone_eviction(ttl: float | None) -> None:
    cache: SlidingCache[str] = SlidingCache(size=3, ttl=ttl)
    for key in ("key1", "key2", "key3"):
        await cache.set(key, key)
    await cache.get("key1")
    cloned = await cache.clone()
    await cloned.get("key2")

    await cache.set("key4", "key4")
    assert not await cache.has("key2")
    assert await cloned.has("key2")
    await cloned.set("key4", "key4")
    assert not await cloned.has("key3")
    assert await cache.has("key3")
    assert await cache.size() == await cloned.size() == 3


@pytest.mark.asyncio
@pytest.mark.unit
async def test_cache_clone_expiration() -> None:
    now = 0.0
    cache: SlidingCache[object] = SlidingCache(size=2, ttl=10)
    cache._items = TTLCache(maxsize=2, ttl=10, timer=lambda: now)
    value = Lock()
    await cache.set("key1", Lock())
    now = 5.0
    await cache.set("key2", value)
    cloned = await cache.clone()

    now = 9.999
    assert await cache.has("key1") and await cloned.has("key1")
    now = 10.0
    partially_expired = await cache.clone()
    for current in (cache, cloned, partially_expired):
        assert not await current.has("key1")
        assert await current.get("key2") is value
        assert await current.size() == 1
    now = 14.999
    for current in (cache, cloned, partially_expired):
        assert await current.get("key2") is value
    now = 15.0
    assert await cache.size() == await cloned.size() == await partially_expired.size() == 0
