# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from beeai_framework.serialization import Serializable


class Counter(Serializable[dict[str, int]]):
    def __init__(self, value: int = 0) -> None:
        self.value = value

    def increment(self) -> None:
        self.value += 1

    def create_snapshot(self) -> dict[str, int]:
        return {"value": self.value}

    def load_snapshot(self, snapshot: dict[str, int]) -> None:
        self.value = snapshot["value"]


def main() -> None:
    counter = Counter(3)
    counter.increment()

    serialized = counter.serialize()
    restored = Counter.from_serialized(serialized)

    print(counter.value)
    print(restored.value)


if __name__ == "__main__":
    main()
