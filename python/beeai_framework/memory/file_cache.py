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


import os
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import Any, Generic, TypeVar

import aiofiles  # type: ignore

from beeai_framework.logger import Logger
from beeai_framework.memory.base_cache import BaseCache
from beeai_framework.memory.serializer import Serializer
from beeai_framework.memory.sliding_cache import SlidingCache

logger = Logger(__name__)

T = TypeVar("T")


def cache() -> Callable[..., Any]:
    """Decorator to cache method results."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        cache_key = f"_cache_{func.__name__}"

        @wraps(func)
        async def wrapper(self: Any, *args: int, **kwargs: int) -> Any:
            if not hasattr(self, cache_key):
                setattr(self, cache_key, await func(self, *args, **kwargs))
            return getattr(self, cache_key)

        wrapper.clear_cache = lambda self: (delattr(self, cache_key) if hasattr(self, cache_key) else None)  # type: ignore
        return wrapper

    return decorator


@dataclass
class Input:
    """Input configuration for FileCache."""

    full_path: str


class FileCache(BaseCache[T], Generic[T]):
    """File-based cache implementation."""

    def __init__(self, input_config: Input) -> None:
        """Initialize the FileCache with the given input configuration."""
        super().__init__()
        self._input = input_config
        self._register()

    @classmethod
    def _register(cls) -> None:
        """Register the cache class."""
        Serializer.register(
            cls,
            {
                "to_plain": lambda x: x.create_snapshot(),
                "from_plain": lambda x: cls.from_snapshot(x),
            },
        )

    @property
    def source(self) -> str:
        """Get the source file path."""
        return self._input.full_path

    @classmethod
    async def from_provider(cls, provider: BaseCache[T], input_config: Input) -> "FileCache[T]":
        """Create a new FileCache instance from a provider."""
        async with aiofiles.open(input_config.full_path, "w") as f:
            serialized = await provider.serialize()  # Await the serialization
            await f.write(serialized)
        return cls(input_config)

    @cache()
    async def _get_provider(self) -> BaseCache[T]:
        """Get the cache provider instance."""
        try:
            exists = os.path.isfile(self._input.full_path)
        except Exception:
            exists = False

        if exists:
            async with aiofiles.open(self._input.full_path) as f:
                serialized = await f.read()

            deserialized = await Serializer.deserialize(serialized)
            target = deserialized["target"]
            snapshot = deserialized["snapshot"]

            target = Serializer.get_factory(target).ref
            instance = target.from_snapshot(snapshot)

            if not isinstance(instance, BaseCache):
                raise TypeError("Provided file does not serialize any instance of BaseCache class.")

            return instance
        else:
            return SlidingCache(size=float("inf"), ttl=float("inf"))

    async def reload(self) -> None:
        """Reload the cache from the file."""
        self._get_provider.clear_cache(self)
        await self._get_provider()

    async def _save(self) -> None:
        """Save the cache to the file."""
        provider = await self._get_provider()
        async with aiofiles.open(self._input.full_path, "w") as f:
            serialized = await provider.serialize()  # Await the serialization
            await f.write(serialized)

    async def size(self) -> int:  # type: ignore
        """Get the number of items in the cache."""
        provider = await self._get_provider()
        return provider.size()  # type: ignore

    async def set(self, key: str, value: T) -> None:
        """Set a value in the cache."""
        provider = await self._get_provider()
        await provider.set(key, value)
        try:
            await provider.get(key)
        finally:
            await self._save()

    async def get(self, key: str) -> T:
        """Get a value from the cache."""
        provider = await self._get_provider()
        return await provider.get(key)  # type: ignore

    async def has(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        provider = await self._get_provider()
        return await provider.has(key)  # type: ignore

    async def delete(self, key: str) -> bool:
        """Delete a key from the cache."""
        provider = await self._get_provider()
        result = await provider.delete(key)
        await self._save()
        return result  # type: ignore

    async def clear(self) -> None:  # type: ignore
        """Clear all items from the cache."""
        provider = await self._get_provider()
        await provider.clear()
        await self._save()

    async def create_snapshot(self) -> dict[str, Any]:
        """Create a serializable snapshot of the current state."""
        return {
            "input": {"full_path": self._input.full_path},
            "provider": await self._get_provider(),
        }

    def load_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Restore state from a snapshot."""
        for key, value in snapshot.items():
            setattr(self, key, value)

    @classmethod
    def from_snapshot(cls, snapshot: dict[str, Any]) -> "FileCache[T]":
        """Create an instance from a snapshot."""
        instance = cls(Input(full_path=snapshot["input"]["full_path"]))
        instance.load_snapshot(snapshot)
        return instance


if __name__ == "__main__":
    import asyncio
    import os
    import tempfile
    from pathlib import Path

    async def test_file_cache() -> None:
        try:
            # Create a temporary directory for our test cache files
            with tempfile.TemporaryDirectory() as temp_dir:
                cache_file = Path(temp_dir) / "test_cache.json"

                logger.info("1. Creating and Testing Basic Cache Operations:")
                # Initialize the cache
                cache = FileCache[str](Input(str(cache_file)))

                # Test basic operations
                logger.info("Setting values in cache...")
                await cache.set("key1", "value1")
                await cache.set("key2", "value2")

                # Verify values
                value1 = await cache.get("key1")
                value2 = await cache.get("key2")
                logger.info(f"Retrieved values: key1={value1}, key2={value2}")

                # Check existence
                has_key = await cache.has("key1")
                logger.info(f"Has key1: {has_key}")

                # Get cache size
                size = await cache.size()
                logger.info(f"Cache size: {size}")

                logger.info("2. Testing File Persistence:")
                # Verify file was created
                logger.info(f"Cache file exists: {cache_file.exists()}")
                logger.info(f"Cache file size: {cache_file.stat().st_size} bytes")

                logger.info("3. Testing Delete Operation:")
                # Delete a key
                deleted = await cache.delete("key2")
                logger.info(f"Deleted key2: {deleted}")
                has_key2 = await cache.has("key2")
                logger.info(f"Has key2 after delete: {has_key2}")

                logger.info("4. Testing Clear Operation:")
                # Clear the cache
                await cache.clear()
                size = await cache.size()
                logger.info(f"Cache size after clear: {size}")

                logger.info("5. Testing Provider Creation:")
                # Test with non-existent file
                new_file = Path(temp_dir) / "new_cache.json"
                new_cache = FileCache[str](Input(str(new_file)))
                await new_cache.set("test_key", "test_value")
                logger.info(f"Created new cache file: {new_file.exists()}")
                logger.info("End of file cache operations")

        except Exception as e:
            logger.error(f"Error during test: {e!s}")

    # Run the test
    asyncio.run(test_file_cache())
