from __future__ import annotations

# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0
import json
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime
from itertools import count
from typing import Any, ClassVar, Generic, TypeVar

T = TypeVar("T")
SerializableT = TypeVar("SerializableT", bound="Serializable[Any]")


class SerializerError(RuntimeError):
    """Raised when serialization or deserialization fails."""


@dataclass(slots=True)
class _SerializerFactory:
    ref: type[Any]
    to_plain: Callable[[Any], Any]
    from_plain: Callable[[Any], Any]
    create_empty: Callable[[], Any] | None = None
    update_instance: Callable[[Any, Any], Any] | None = None


class Serializer:
    """Minimal registry driven serializer compatible with the documentation examples."""

    _factories: ClassVar[dict[str, _SerializerFactory]] = {}
    _type_to_name: ClassVar[dict[type[Any], str]] = {}
    version: ClassVar[str] = "0.0.0"

    @staticmethod
    def _class_name(ref: type[Any]) -> str:
        return ref.__name__

    @classmethod
    def has_factory(cls, name: str) -> bool:
        return name in cls._factories

    @classmethod
    def has_factory_for_type(cls, ref: type[Any]) -> bool:
        return ref in cls._type_to_name

    @classmethod
    def register(
        cls,
        ref: type[Any],
        *,
        to_plain: Callable[[Any], Any],
        from_plain: Callable[[Any], Any],
        create_empty: Callable[[], Any] | None = None,
        update_instance: Callable[[Any, Any], Any] | None = None,
        aliases: Iterable[str] | None = None,
    ) -> None:
        """Register custom serialization logic for a class."""
        name = cls._class_name(ref)
        factory = _SerializerFactory(
            ref=ref,
            to_plain=to_plain,
            from_plain=from_plain,
            create_empty=create_empty,
            update_instance=update_instance,
        )

        cls._factories[name] = factory
        cls._type_to_name[ref] = name

        if aliases:
            for alias in aliases:
                cls._factories[alias] = factory

    @classmethod
    def register_serializable(
        cls,
        ref: type[Serializable[Any]],
        *,
        aliases: Iterable[str] | None = None,
    ) -> None:
        """Register classes that implement the Serializable protocol."""

        if cls.has_factory_for_type(ref):
            return

        cls.register(
            ref,
            to_plain=lambda instance: instance.create_snapshot(),
            from_plain=lambda snapshot: ref.from_snapshot(snapshot),
            create_empty=lambda: ref.__new__(ref),  # type: ignore[misc]
            update_instance=lambda instance, snapshot: instance.load_snapshot(snapshot),
            aliases=aliases,
        )

    @classmethod
    def deregister(cls, ref: type[Any]) -> None:
        """Remove class registration."""
        name = cls._type_to_name.pop(ref, None)
        if name is None:
            return

        for key in [alias for alias, factory in cls._factories.items() if factory.ref is ref]:
            cls._factories.pop(key, None)

    @classmethod
    def ensure_registered(cls, ref: type[Any]) -> None:
        if cls.has_factory_for_type(ref):
            return

        # Avoid circular import by referencing Serializable at runtime
        if issubclass(ref, Serializable):
            cls.register_serializable(ref)
            return

        raise SerializerError(f'Class "{ref.__name__}" is not registered with the serializer.')

    @classmethod
    def serialize(cls, raw_data: Any) -> str:
        """Serialize python objects into a JSON string."""
        ref_counter = count(1)
        seen: dict[int, str] = {}
        payload = {
            "__version": cls.version,
            "__root": cls._encode(raw_data, seen, ref_counter),
        }
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)

    @classmethod
    def deserialize(
        cls,
        data: str | Mapping[str, Any],
        *,
        expected_type: type[T] | None = None,
        extra_classes: Iterable[type[Any]] | None = None,
    ) -> T | Any:
        """Deserialize objects previously produced by `serialize`."""
        if isinstance(data, str):
            try:
                payload = json.loads(data)
            except json.JSONDecodeError as exc:
                raise SerializerError("Invalid serialized payload.") from exc
        else:
            payload = dict(data)

        if not isinstance(payload, dict) or "__root" not in payload:
            raise SerializerError("Serialized payload is missing the '__root' key.")

        if extra_classes:
            for extra in extra_classes:
                if isinstance(extra, type):
                    registrar = getattr(extra, "register", None)
                    if callable(registrar):
                        registrar()
                    try:
                        cls.ensure_registered(extra)
                    except SerializerError:
                        # Allow extra classes that register themselves on demand.
                        continue

        seen: dict[str, Any] = {}
        value = cls._decode(payload["__root"], seen)

        if expected_type is not None and not isinstance(value, expected_type):
            raise SerializerError(
                f"Deserialized value is of type {type(value).__name__}, expected {expected_type.__name__}.",
            )

        return value

    @classmethod
    def _find_factory_for_instance(cls, value: Any) -> tuple[_SerializerFactory, str]:
        cls.ensure_registered(type(value))

        for candidate in type(value).__mro__:
            name = cls._type_to_name.get(candidate)
            if name and name in cls._factories:
                return cls._factories[name], name

        raise SerializerError(f'No serializer factory found for "{type(value).__name__}".')

    @classmethod
    def _encode(cls, value: Any, seen: dict[int, str], ref_counter: Iterator[int]) -> Any:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value

        if isinstance(value, (list, tuple, set)):
            return [cls._encode(item, seen, ref_counter) for item in value]

        if isinstance(value, dict):
            return {str(key): cls._encode(item, seen, ref_counter) for key, item in value.items()}

        factory, name = cls._find_factory_for_instance(value)
        obj_id = id(value)
        if obj_id in seen:
            return {
                "__serializer": True,
                "__ref": seen[obj_id],
            }

        ref_id = str(next(ref_counter))
        seen[obj_id] = ref_id

        plain = factory.to_plain(value)
        encoded_plain = cls._encode(plain, seen, ref_counter)
        return {
            "__serializer": True,
            "__class": name,
            "__ref": ref_id,
            "__value": encoded_plain,
        }

    @classmethod
    def _decode(cls, value: Any, seen: dict[str, Any]) -> Any:
        if isinstance(value, list):
            return [cls._decode(item, seen) for item in value]

        if isinstance(value, dict):
            if value.get("__serializer") is True:
                ref_id = value.get("__ref")
                class_name = value.get("__class")

                if class_name is None:
                    if ref_id is None or ref_id not in seen:
                        raise SerializerError("Encountered reference to unknown object.")
                    return seen[ref_id]

                if class_name not in cls._factories:
                    raise SerializerError(f'Class "{class_name}" was not registered.')

                factory = cls._factories[class_name]
                payload = value.get("__value")

                if ref_id is not None and ref_id in seen:
                    instance = seen[ref_id]
                    if payload is not None and factory.update_instance:
                        decoded_payload = cls._decode(payload, seen)
                        factory.update_instance(instance, decoded_payload)
                    return instance

                if factory.create_empty and factory.update_instance:
                    instance = factory.create_empty()
                    if ref_id is not None:
                        seen[ref_id] = instance
                    decoded_payload = cls._decode(payload, seen) if payload is not None else None
                    factory.update_instance(instance, decoded_payload)
                    return instance

                decoded_payload = cls._decode(payload, seen) if payload is not None else None
                instance = factory.from_plain(decoded_payload)
                if ref_id is not None:
                    seen[ref_id] = instance
                return instance

            return {key: cls._decode(item, seen) for key, item in value.items()}

        return value


class Serializable(Generic[T]):
    """Mixin that provides convenience helpers for registering serializable classes."""

    def __init_subclass__(
        cls,
        *,
        auto_register: bool = True,
        aliases: Iterable[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init_subclass__(**kwargs)
        if auto_register:
            Serializer.register_serializable(cls, aliases=aliases)

    def serialize(self) -> str:
        return Serializer.serialize(self)

    @classmethod
    def register(cls, *, aliases: Iterable[str] | None = None) -> None:
        Serializer.register_serializable(cls, aliases=aliases)

    @classmethod
    def from_serialized(
        cls: type[SerializableT],
        data: str,
        *,
        extra_classes: Iterable[type[Any]] | None = None,
    ) -> SerializableT:
        value = Serializer.deserialize(data, expected_type=cls, extra_classes=extra_classes)
        assert isinstance(value, cls)
        return value

    @classmethod
    def from_snapshot(cls: type[SerializableT], snapshot: T) -> SerializableT:
        instance = cls.__new__(cls)  # type: ignore[misc]
        instance.load_snapshot(snapshot)
        return instance

    def to_snapshot(self) -> T:
        return self.create_snapshot()

    def create_snapshot(self) -> T:
        raise NotImplementedError

    def load_snapshot(self, snapshot: T) -> None:
        raise NotImplementedError


Serializer.register(
    datetime,
    to_plain=lambda value: value.isoformat(),
    from_plain=lambda data: datetime.fromisoformat(data),
)
