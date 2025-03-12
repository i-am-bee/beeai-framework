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


import time
from collections import OrderedDict
from typing import Any, Generic, TypeVar

from beeai_framework.memory.base_cache import BaseCache
from beeai_framework.memory.serializer import Serializer

T = TypeVar("T")


class SlidingCache(BaseCache[T], Generic[T]):
    """Cache implementation using a sliding window strategy."""

    def __init__(self, size: float = float("inf"), ttl: float | None = None) -> None:
        """
        Initialize the sliding cache.

        Args:
            size: Maximum number of items (default: infinite)
            ttl: Time-to-live in seconds (default: None)
        """
        super().__init__()
        self._max_size = size
        self._ttl = ttl
        self._items: OrderedDict[str, tuple[T, float]] = OrderedDict()
        # Register for serialization
        self._register()

    @classmethod
    def _register(cls) -> None:
        """Register this class for serialization."""
        Serializer.register_serializable(cls)

    def _evict_expired(self) -> None:
        """Remove expired entries."""
        if self._ttl is None:
            return

        current_time = time.time()
        expired_keys = [key for key, (_, timestamp) in self._items.items() if current_time - timestamp > self._ttl]

        for key in expired_keys:
            self._items.pop(key, None)

    def _evict_overflow(self) -> None:
        """Remove oldest entries if size limit is exceeded."""
        while len(self._items) > self._max_size:
            self._items.popitem(last=False)

    async def set(self, key: str, value: T) -> None:
        """Set a value in the cache."""
        self._evict_expired()
        self._items[key] = (value, time.time())
        self._evict_overflow()

    async def get(self, key: str) -> T | None:
        """Get a value from the cache."""
        self._evict_expired()
        if key in self._items:
            value, _ = self._items[key]
            # Move to end (most recently used)
            self._items.move_to_end(key)
            return value
        return None

    async def has(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        self._evict_expired()
        return key in self._items

    async def delete(self, key: str) -> bool:
        """Delete a key from the cache."""
        if key in self._items:
            del self._items[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all items from the cache."""
        self._items.clear()

    def size(self) -> int:
        """Get the current number of items in the cache."""
        self._evict_expired()
        return len(self._items)

    async def create_snapshot(self) -> dict[str, Any]:
        """Create a serializable snapshot of the current state."""
        self._evict_expired()
        return {
            "max_size": self._max_size,
            "ttl": self._ttl,
            "items": [(k, v[0], v[1]) for k, v in self._items.items()],
        }

    def load_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Restore state from a snapshot."""
        self._max_size = snapshot["max_size"]
        self._ttl = snapshot["ttl"]
        self._items = OrderedDict()
        for key, value, timestamp in snapshot["items"]:
            self._items[key] = (value, timestamp)

    @classmethod
    def from_snapshot(cls, snapshot: dict[str, Any]) -> "SlidingCache[T]":
        """Create an instance from a snapshot."""
        instance = cls(size=snapshot.get("max_size", float("inf")), ttl=snapshot.get("ttl"))
        instance.load_snapshot(snapshot)
        return instance
