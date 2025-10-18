# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pytest

from beeai_framework.backend import AssistantMessage, UserMessage
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.serialization import Serializable, Serializer


def test_datetime_roundtrip() -> None:
    original = datetime(2024, 1, 1, tzinfo=UTC)

    payload = Serializer.serialize(original)
    restored = Serializer.deserialize(payload, expected_type=datetime)

    assert restored == original


@pytest.mark.asyncio
async def test_unconstrained_memory_roundtrip() -> None:
    memory = UnconstrainedMemory()
    await memory.add(UserMessage("Hello"))
    await memory.add(AssistantMessage("Hi there"))

    payload = memory.serialize()
    restored = UnconstrainedMemory.from_serialized(payload)

    assert len(restored.messages) == 2
    assert restored.messages[0].text == "Hello"


def test_custom_registration() -> None:
    @dataclass
    class ApiToken:
        value: str
        expires_at: datetime

    Serializer.register(
        ApiToken,
        to_plain=lambda token: {
            "value": token.value,
            "expires_at": token.expires_at,
        },
        from_plain=lambda payload: ApiToken(
            value=payload["value"],
            expires_at=payload["expires_at"],
        ),
    )

    token = ApiToken("secret", datetime(2025, 1, 1, tzinfo=UTC))
    payload = Serializer.serialize(token)
    restored = Serializer.deserialize(payload, expected_type=ApiToken)

    assert restored == token


def test_serializable_subclass_roundtrip() -> None:
    class Counter(Serializable[dict[str, int]]):
        def __init__(self, value: int = 0) -> None:
            self.value = value

        def create_snapshot(self) -> dict[str, int]:
            return {"value": self.value}

        def load_snapshot(self, snapshot: dict[str, int]) -> None:
            self.value = snapshot["value"]

    counter = Counter(7)
    payload = counter.serialize()
    restored = Counter.from_serialized(payload)

    assert restored.value == 7


@pytest.mark.asyncio
async def test_deserialize_with_extra_classes() -> None:
    memory = UnconstrainedMemory()
    await memory.add(UserMessage("Where are we?"))

    payload = memory.serialize()

    Serializer.deregister(UserMessage)
    try:
        restored = UnconstrainedMemory.from_serialized(payload, extra_classes=[UserMessage])
    finally:
        UserMessage.register()

    assert len(restored.messages) == 1
    assert isinstance(restored.messages[0], UserMessage)


def test_shared_reference_roundtrip() -> None:
    class Node(Serializable[dict[str, Any]]):
        def __init__(self, name: str) -> None:
            self.name = name
            self.children: list[Node] = []

        def create_snapshot(self) -> dict[str, Any]:
            return {"name": self.name, "children": self.children}

        def load_snapshot(self, snapshot: dict[str, Any]) -> None:
            self.name = snapshot["name"]
            self.children = snapshot["children"]

    child = Node("child")
    root = Node("root")
    root.children = [child, child]

    payload = Serializer.serialize(root)
    restored = Serializer.deserialize(payload, expected_type=Node)

    assert restored.children[0] is restored.children[1]


def test_cyclic_graph_roundtrip() -> None:
    class Ring(Serializable[dict[str, Any]]):
        def __init__(self, name: str) -> None:
            self.name = name
            self.next: Ring | None = None

        def create_snapshot(self) -> dict[str, Any]:
            return {"name": self.name, "next": self.next}

        def load_snapshot(self, snapshot: dict[str, Any]) -> None:
            self.name = snapshot["name"]
            self.next = snapshot["next"]

    node = Ring("ring")
    node.next = node

    payload = Serializer.serialize(node)
    restored = Serializer.deserialize(payload, expected_type=Ring)

    assert restored.next is restored
