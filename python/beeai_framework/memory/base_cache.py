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


from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any, Generic, TypeVar

from beeai_framework.memory.serializer import Serializable

T = TypeVar("T")


class BaseCache(Serializable, ABC, Generic[T]):
    """Abstract base class for all Cache implementations."""

    def __init__(self) -> None:
        """Initialize the cache with an empty elements dictionary."""
        self._elements: dict[str, Any] = {}
        self._enabled: bool = True

    @property
    def enabled(self) -> bool:
        """
        Property that indicates if the cache is enabled.

        Returns:
            bool: True if cache is enabled, False otherwise
        """
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """
        Set the enabled status of the cache.

        Args:
            value (bool): The new enabled status
        """
        self._enabled = value

    @property
    def elements(self) -> dict[str, Any]:
        """
        Property that provides access to the internal elements dictionary.

        Returns:
            Dict[str, Any]: The cache elements
        """
        return self._elements

    async def serialize(self) -> str:
        """Serialize the cache state."""
        snapshot = await self.create_snapshot()
        from beeai_framework.memory.serializer import (
            Serializer,  # Import here to avoid circular imports
        )

        return await Serializer.serialize(
            {
                "target": {
                    "module": self.__class__.__module__,
                    "name": self.__class__.__name__,
                },
                "snapshot": snapshot,
            }
        )

    @abstractmethod
    async def set(self, key: str, value: Any) -> None:
        """Add a element in the cache."""
        pass

    @abstractmethod
    async def get(self, key: str) -> Any:
        """Get a element in the cache."""
        pass

    @abstractmethod
    async def has(self, key: str) -> bool:
        """Get a element in the cache."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete a element in the Cache."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all the Cache content."""
        pass

    def size(self) -> int:
        """Clear all the Cache content."""
        return len(self.elements)

    def is_empty(self) -> bool:
        """Check if memory is empty."""
        return len(self.elements) == 0

    def __iter__(self) -> Iterator[str]:
        return iter(self.elements)

    @abstractmethod
    def create_snapshot(self) -> Any:
        """Create a serializable snapshot of current state."""
        pass

    @abstractmethod
    def load_snapshot(self, state: Any) -> None:
        """Restore state from a snapshot."""
        pass
