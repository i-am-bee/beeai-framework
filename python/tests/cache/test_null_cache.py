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


import pytest
import pytest_asyncio

from beeai_framework.cache.null_cache import NullCache


@pytest_asyncio.fixture
async def cache() -> NullCache[str]:
    _cache: NullCache[str] = NullCache()
    await _cache.set("key1", "value1")
    await _cache.set("key2", "value2")
    await _cache.set("key3", "value3")
    return _cache


@pytest.mark.asyncio
@pytest.mark.unit
async def test_cache_size(cache: NullCache[str]) -> None:
    assert cache.enabled is False
    assert await cache.size() == 0


@pytest.mark.asyncio
@pytest.mark.unit
async def test_cache_set(cache: NullCache[str]) -> None:
    await cache.set("key4", "value4")
    await cache.set("key5", "value5")

    assert await cache.size() == 0


@pytest.mark.asyncio
@pytest.mark.unit
async def test_cache_get(cache: NullCache[str]) -> None:
    value0 = await cache.get("key0")
    value2 = await cache.get("key2")

    assert value0 is None
    assert value2 is None

    assert await cache.size() == 0


@pytest.mark.asyncio
@pytest.mark.unit
async def test_cache_has(cache: NullCache[str]) -> None:
    assert await cache.has("key1") is False
    assert await cache.has("key4") is False


@pytest.mark.asyncio
@pytest.mark.unit
async def test_cache_delete(cache: NullCache[str]) -> None:
    del0 = await cache.delete("key0")
    del2 = await cache.delete("key2")

    assert del0 is True
    assert del2 is True
    assert await cache.size() == 0


@pytest.mark.asyncio
@pytest.mark.unit
async def test_cache_clear(cache: NullCache[str]) -> None:
    assert await cache.size() == 0
    await cache.clear()
    assert await cache.size() == 0
