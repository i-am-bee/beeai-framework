import asyncio
import json
import sys
import tempfile
import time
import traceback
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Generic, TypeVar

from beeai_framework.cache import BaseCache
from beeai_framework.errors import FrameworkError

T = TypeVar("T")


class JsonFileCache(BaseCache[T], Generic[T]):
    """Simple file-backed cache with optional LRU eviction and TTL support."""

    def __init__(self, path: Path, *, size: int = 128, ttl: float | None = None) -> None:
        super().__init__()
        self._path = path
        self._size = size
        self._ttl = ttl
        self._items: OrderedDict[str, tuple[T, float | None]] = OrderedDict()
        self._load_from_disk()

    @property
    def source(self) -> Path:
        return self._path

    @classmethod
    async def from_mapping(
        cls,
        path: Path,
        items: Mapping[str, T],
        *,
        size: int = 128,
        ttl: float | None = None,
    ) -> "JsonFileCache[T]":
        cache = cls(path, size=size, ttl=ttl)
        for key, value in items.items():
            await cache.set(key, value)
        return cache

    async def size(self) -> int:
        await self._purge_expired()
        return len(self._items)

    async def set(self, key: str, value: T) -> None:
        await self._purge_expired()
        expires_at = time.time() + self._ttl if self._ttl is not None else None
        if key in self._items:
            self._items.pop(key)
        self._items[key] = (value, expires_at)
        await self._enforce_capacity()
        self._dump_to_disk()

    async def get(self, key: str) -> T | None:
        await self._purge_expired()
        if key not in self._items:
            return None

        value, expires_at = self._items.pop(key)
        self._items[key] = (value, expires_at)
        return value

    async def has(self, key: str) -> bool:
        await self._purge_expired()
        return key in self._items

    async def delete(self, key: str) -> bool:
        await self._purge_expired()
        if key not in self._items:
            return False

        self._items.pop(key)
        self._dump_to_disk()
        return True

    async def clear(self) -> None:
        self._items.clear()
        if self._path.exists():
            self._path.unlink()

    async def reload(self) -> None:
        self._items.clear()
        self._load_from_disk()
        await self._purge_expired()

    async def _purge_expired(self) -> None:
        now = time.time()
        expired_keys = [
            key for key, (_, expires_at) in list(self._items.items()) if expires_at is not None and expires_at <= now
        ]
        for key in expired_keys:
            self._items.pop(key, None)
        if expired_keys:
            self._dump_to_disk()

    async def _enforce_capacity(self) -> None:
        while len(self._items) > self._size:
            oldest_key, _ = self._items.popitem(last=False)

    def _load_from_disk(self) -> None:
        if not self._path.exists():
            return

        try:
            raw = json.loads(self._path.read_text())
        except json.JSONDecodeError:
            return

        now = time.time()
        for key, payload in raw.items():
            expires_at = payload.get("expires_at")
            if expires_at is not None and expires_at <= now:
                continue
            self._items[key] = (payload["value"], expires_at)

    def _dump_to_disk(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {key: {"value": value, "expires_at": expires_at} for key, (value, expires_at) in self._items.items()}
        self._path.write_text(json.dumps(data, indent=2))


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "bee_cache.json"
        cache: JsonFileCache[dict[str, str]] = JsonFileCache(path, size=2, ttl=1.5)

        await cache.set("profile", {"name": "Bee", "role": "assistant"})
        await cache.set("settings", {"theme": "dark"})
        print(f"Cache persisted to {cache.source}")

        await cache.set("session", {"token": "abc123"})
        print(await cache.has("profile"))  # False -> evicted when capacity exceeded

        reloaded: JsonFileCache[dict[str, str]] = JsonFileCache(path, size=2, ttl=1.5)
        print(await reloaded.get("settings"))  # {'theme': 'dark'}

        await asyncio.sleep(1.6)
        await reloaded.reload()
        print(await reloaded.get("session"))  # None -> TTL expired


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
