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


from typing import Any, Generic, TypeVar

from beeai_framework.memory.base_cache import BaseCache
from beeai_framework.memory.serializer import Serializer

T = TypeVar("T")


class UnconstrainedCache(BaseCache[T], Generic[T]):
    """Cache implementation without size or time constraints."""

    def __init__(self) -> None:
        """Initialize the unconstrained cache."""
        super().__init__()
        self._provider: dict[str, T] = {}
        self._register()

    @classmethod
    def _register(cls) -> None:
        """Register for serialization."""
        Serializer.register(
            cls,
            {
                "to_plain": lambda x: {
                    "enabled": x.enabled,
                    "provider": dict(x._provider),
                },
                "from_plain": lambda x: cls.from_snapshot(x),
            },
        )

    async def get(self, key: str) -> T | None:
        """Get a value from the cache."""
        return self._provider.get(key)

    async def has(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        return key in self._provider

    def clear(self) -> None:
        """Clear all items from the cache."""
        self._provider.clear()

    async def delete(self, key: str) -> bool:
        """Delete a key from the cache."""
        if key in self._provider:
            del self._provider[key]
            return True
        return False

    async def set(self, key: str, value: T) -> None:
        """Set a value in the cache."""
        self._provider[key] = value

    def size(self) -> int:
        """Get the current number of items in the cache."""
        return len(self._provider)

    async def create_snapshot(self) -> dict[str, Any]:
        """Create a serializable snapshot of the current state."""
        return {"enabled": self.enabled, "provider": dict(self._provider)}

    def load_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Restore state from a snapshot."""
        self._enabled = snapshot["enabled"]
        self._provider = dict(snapshot["provider"])

    @classmethod
    def from_snapshot(cls, snapshot: dict[str, Any]) -> "UnconstrainedCache[T]":
        """Create an instance from a snapshot."""
        instance = cls()
        instance.load_snapshot(snapshot)
        return instance


if __name__ == "__main__":
    import asyncio

    async def test_unconstrained_cache() -> None:
        try:
            print("\n1. Testing Basic Operations:")
            # Create cache instance
            cache = UnconstrainedCache[str]()

            # Test setting and getting values
            print("Setting test values...")
            await cache.set("key1", "value1")
            await cache.set("key2", "value2")
            await cache.set("key3", "value3")

            # Test retrieval
            value1 = await cache.get("key1")
            value2 = await cache.get("key2")
            print(f"Retrieved values: key1={value1}, key2={value2}")

            # Test has method
            exists = await cache.has("key1")
            not_exists = await cache.has("nonexistent")
            print(f"Has key1: {exists}")
            print(f"Has nonexistent: {not_exists}")

            # Test size
            size = cache.size()
            print(f"Cache size: {size}")

            print("\n2. Testing Delete Operation:")
            # Test deletion
            deleted = await cache.delete("key2")
            size_after_delete = cache.size()
            print(f"Deleted key2: {deleted}")
            print(f"Size after delete: {size_after_delete}")

            print("\n3. Testing Clear Operation:")
            # Test clear
            cache.clear()
            size_after_clear = cache.size()
            print(f"Size after clear: {size_after_clear}")

            print("\n4. Testing Serialization:")
            # Test serialization
            new_cache = UnconstrainedCache[str]()
            await new_cache.set("test1", "data1")
            await new_cache.set("test2", "data2")

            # Create snapshot
            snapshot = await new_cache.create_snapshot()
            print(f"Created snapshot: {snapshot}")

            # Create new instance from snapshot
            restored_cache = UnconstrainedCache[str].from_snapshot(snapshot)
            restored_value = await restored_cache.get("test1")
            print(f"Restored value from snapshot: {restored_value}")

            print("\n5. Testing Enabled Property:")
            # Test enabled property
            original_state = new_cache.enabled
            new_cache.enabled = False
            print(f"Original enabled state: {original_state}")
            print(f"New enabled state: {new_cache.enabled}")

            print("\n6. Testing Large Dataset:")
            # Test with larger dataset
            large_cache = UnconstrainedCache[int]()
            print("Adding 1000 items...")
            for i in range(1000):
                await large_cache.set(f"key{i}", i)

            large_size = large_cache.size()
            sample_value = await large_cache.get("key500")
            print(f"Large cache size: {large_size}")
            print(f"Sample value (key500): {sample_value}")

            print("\nAll tests completed successfully!")

        except Exception as e:
            print(f"Error during test: {e!s}")

    # Run the tests
    asyncio.run(test_unconstrained_cache())
