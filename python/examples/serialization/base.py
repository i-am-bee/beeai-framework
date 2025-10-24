# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from datetime import UTC, datetime

from beeai_framework.serialization import Serializer


def main() -> None:
    original = datetime(2024, 1, 1, tzinfo=UTC)
    serialized = Serializer.serialize(original)
    restored = Serializer.deserialize(serialized, expected_type=datetime)

    print(serialized)
    print(restored.isoformat())
    print(restored == original)


if __name__ == "__main__":
    main()
