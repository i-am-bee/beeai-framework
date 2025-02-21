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
from typing import Any, ClassVar, TypeVar

T = TypeVar("T", bound="Serializable")


class Serializable(ABC):
    """Base class for all serializable objects."""

    _registered_classes: ClassVar[dict[str, type["Serializable"]]] = {}

    def __init_subclass__(cls, **kwargs: int) -> None:
        """Automatically register subclasses when they're created."""
        super().__init_subclass__(**kwargs)
        cls._registered_classes[cls.__name__] = cls

    @classmethod
    def register(cls, aliases: list[str] | None = None) -> None:
        """Register the class and any aliases for serialization."""
        cls._registered_classes[cls.__name__] = cls
        if aliases:
            for alias in aliases:
                if alias in cls._registered_classes and cls._registered_classes[alias] != cls:
                    raise ValueError(f"Alias '{alias}' already registered to a different class")
                cls._registered_classes[alias] = cls

    @classmethod
    def from_serialized(cls: type[T], data: dict[str, Any]) -> T:
        """Create an instance from serialized data."""
        instance = cls.__new__(cls)
        Serializable.__init__(instance)
        instance.load_snapshot(data)
        return instance

    @classmethod
    def from_snapshot(cls: type[T], data: dict[str, Any]) -> T:
        """Create an instance from a snapshot."""
        instance = cls.__new__(cls)
        Serializable.__init__(instance)
        instance.load_snapshot(data)
        return instance

    def serialize(self) -> dict[str, Any]:
        """Serialize the object to a dictionary."""
        return {"__class": self.__class__.__name__, "__value": self.create_snapshot()}

    @abstractmethod
    def create_snapshot(self) -> dict[str, Any]:
        """Create a serializable snapshot of the object's state."""
        raise NotImplementedError

    @abstractmethod
    def load_snapshot(self, state: dict[str, Any]) -> None:
        """Load object state from a snapshot."""
        raise NotImplementedError

    def clone(self: T) -> T:
        """Create a deep copy of the object."""
        snapshot = self.create_snapshot()
        return type(self).from_snapshot(deepcopy(snapshot))


# Example of how to use the base class:
class ExampleSerializable(Serializable):
    def __init__(self, data: str) -> None:
        super().__init__()
        self.data = data

    @classmethod
    def register(cls, aliases: list[str] | None = None) -> None:
        """Register with custom aliases."""
        super().register(aliases)

    def create_snapshot(self) -> dict[str, Any]:
        return {"data": self.data}

    def load_snapshot(self, state: dict[str, Any]) -> None:
        self.data = state["data"]


# Usage example:
if __name__ == "__main__":
    # Register the class with an alias
    ExampleSerializable.register(aliases=["Example", "ExampleClass"])

    # Create and serialize an instance
    obj = ExampleSerializable("test data")
    serialized = obj.serialize()

    # Create new instance from serialized data
    new_obj = ExampleSerializable.from_serialized(serialized["__value"])

    # Create a clone
    cloned = obj.clone()
