# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from datetime import UTC, datetime

from beeai_framework.serialization import Serializer


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
    from_plain=lambda payload, ref: ref(
        value=payload["value"],
        expires_at=payload["expires_at"],
    ),
)


def main() -> None:
    token = ApiToken("example-token", datetime(2025, 1, 1, tzinfo=UTC))
    serialized = Serializer.serialize(token)
    restored = Serializer.deserialize(serialized, expected_type=ApiToken)

    print(restored)
    print(restored.expires_at.isoformat())


if __name__ == "__main__":
    main()
