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
from copy import deepcopy
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound="Serializable")


class Serializable(ABC):
    """Base class for serializable objects."""

    @classmethod
    async def from_snapshot(cls: type[T], snapshot: dict[str, Any]) -> T:
        """Create instance from snapshot."""
        if issubclass(cls, BaseModel):
            return cls(**dict(snapshot))
        else:
            instance = cls()
            await instance.load_snapshot(snapshot)
            return instance

    async def clone(self: T) -> T:
        """Create a deep copy of the object."""
        snapshot: dict[str, Any] = await self.create_snapshot()
        return await type(self).from_snapshot(deepcopy(snapshot))

    @abstractmethod
    async def create_snapshot(self) -> dict[str, Any]:
        """Create serializable snapshot."""
        pass

    @abstractmethod
    async def load_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Restore from snapshot."""
        pass
