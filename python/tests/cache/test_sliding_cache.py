# Copyright 2025 IBM Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import asyncio

import pytest
import pytest_asyncio

from beeai_framework.cache.sliding_cache import SlidingCache


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
