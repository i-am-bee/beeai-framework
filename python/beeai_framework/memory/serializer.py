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
import base64
import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, TypeVar

from beeai_framework.memory.errors import SerializerError

T = TypeVar("T", bound="Serializable")


class SerializerFactory:
    """Factory for serializable class registration and instantiation."""

    def __init__(self, cls_ref: type["Serializable"]) -> None:
        self.ref = cls_ref
        self.module = cls_ref.__module__
        self.name = cls_ref.__name__
        self.to_plain: Callable[..., Any] | None = None
        self.from_plain: Callable[..., Any] | None = None
        self.create_empty: Callable[..., Any] | None = None
        self.update_instance: Callable[..., Any] | None = None


class Serializable(ABC):
    """Base class for serializable objects."""

    @classmethod
    def register(cls) -> None:
        """Register for serialization."""
        Serializer.register_serializable(cls)

    @classmethod
    def from_snapshot(cls: type[T], snapshot: dict[str, Any]) -> T:
        """Create instance from snapshot."""
        instance = cls()
        instance.load_snapshot(snapshot)
        return instance

    @abstractmethod
    async def create_snapshot(self) -> dict[str, Any]:
        """Create serializable snapshot."""
        pass

    @abstractmethod
    def load_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Restore from snapshot."""
        pass


class Serializer:
    """Main serializer class."""

    _factories: dict[str, SerializerFactory] = {}  # noqa: RUF012

    @classmethod
    def register_serializable(cls, target_cls: type[Serializable]) -> None:
        """Register a serializable class."""
        # Register with both module name and __main__
        names = [
            f"{target_cls.__module__}.{target_cls.__name__}",
            f"__main__.{target_cls.__name__}",
        ]
        factory = SerializerFactory(target_cls)
        factory.to_plain = lambda x: x.create_snapshot()
        factory.from_plain = target_cls.from_snapshot

        for name in names:
            cls._factories[name] = factory

    @classmethod
    def register(cls, target_cls: type[Any], processors: dict[str, Callable[..., Any]]) -> None:
        """Register a class with custom processors."""
        names = [
            f"{target_cls.__module__}.{target_cls.__name__}",
            f"__main__.{target_cls.__name__}",
        ]
        factory = SerializerFactory(target_cls)
        factory.to_plain = processors.get("to_plain")
        factory.from_plain = processors.get("from_plain")
        factory.create_empty = processors.get("create_empty")
        factory.update_instance = processors.get("update_instance")

        for name in names:
            cls._factories[name] = factory

    @classmethod
    def get_factory(cls, class_name: str) -> SerializerFactory:
        """Get factory for class name."""
        factory = cls._factories.get(class_name)
        if not factory:
            raise SerializerError(f"Class {class_name} not registered")
        return factory

    @classmethod
    async def serialize(cls, data: Any) -> str:
        """Serialize data to JSON string with async support."""

        async def serialize_obj(obj: Any) -> Any:
            if isinstance(obj, str | int | float | bool) or obj is None:
                return obj

            if isinstance(obj, list | tuple):
                return [await serialize_obj(item) for item in obj]

            if isinstance(obj, dict):
                return {str(k): await serialize_obj(v) for k, v in obj.items()}

            class_name = f"{obj.__class__.__module__}.{obj.__class__.__name__}"
            try:
                factory = cls.get_factory(class_name)
                if factory.to_plain:
                    snapshot = await obj.create_snapshot() if hasattr(obj, "create_snapshot") else factory.to_plain(obj)
                    return {
                        "__type": class_name,
                        "__value": await serialize_obj(snapshot),
                    }
            except SerializerError:
                pass

            raise SerializerError(f"Cannot serialize object of type {class_name}")

        serialized_data = await serialize_obj(data)
        return json.dumps(serialized_data)

    @classmethod
    async def deserialize(cls, data: str) -> Any:
        """Deserialize JSON string to object with async support."""

        async def deserialize_obj(obj: Any) -> Any:
            if isinstance(obj, str | int | float | bool) or obj is None:
                return obj

            if isinstance(obj, list):
                return [await deserialize_obj(item) for item in obj]

            if isinstance(obj, dict):
                if "__type" in obj:
                    factory = cls.get_factory(obj["__type"])
                    if factory.from_plain:
                        return factory.from_plain(await deserialize_obj(obj["__value"]))
                return {k: await deserialize_obj(v) for k, v in obj.items()}

            return obj

        return await deserialize_obj(json.loads(data))


# Register basic types
for type_cls in (list, dict, set):
    Serializer.register(
        type_cls,
        {
            "to_plain": lambda x: list(x) if isinstance(x, list | set) else dict(x),
            "from_plain": lambda x: type_cls(x),  # noqa: B023
        },
    )

Serializer.register(
    datetime,
    {
        "to_plain": lambda x: x.isoformat(),
        "from_plain": lambda x: datetime.fromisoformat(x),
    },
)

Serializer.register(
    bytes,
    {
        "to_plain": lambda x: base64.b64encode(x).decode("utf-8"),
        "from_plain": lambda x: base64.b64decode(x.encode("utf-8")),
    },
)

if __name__ == "__main__":

    class User(Serializable):
        def __init__(self, name: str = "", age: int = 0, email: str | None = None) -> None:
            self.name = name
            self.age = age
            self.email = email

        async def create_snapshot(self) -> dict[str, Any]:
            return {"name": self.name, "age": self.age, "email": self.email}

        def load_snapshot(self, snapshot: dict[str, Any]) -> None:
            self.name = snapshot["name"]
            self.age = snapshot["age"]
            self.email = snapshot.get("email")

    # Register the class
    User.register()

    async def main() -> None:
        try:
            # Create and test serialization
            user = User("Alice", 30, "alice@example.com")

            print("\n1. Basic User Serialization:")
            serialized = await Serializer.serialize(user)
            print(f"Serialized User: {serialized}")

            deserialized = await Serializer.deserialize(serialized)
            print(f"Deserialized User: {deserialized.name}, {deserialized.age}")

            # Test with built-in types
            print("\n2. Built-in Types Serialization:")
            data = {"user": user, "numbers": [1, 2, 3], "timestamp": datetime.now(tz=UTC)}

            serialized = await Serializer.serialize(data)
            print(f"Serialized Data: {serialized}")

            deserialized = await Serializer.deserialize(serialized)
            print(f"Deserialized Data (user name): {deserialized['user'].name}")

        except SerializerError as e:
            print(f"Serialization Error: {e}")
        except Exception as e:
            print(f"Unexpected Error: {e!s}")

    # Run the async main function
    asyncio.run(main())
