# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from beeai_framework.backend import AssistantMessage, UserMessage
from beeai_framework.memory import UnconstrainedMemory

SERIALIZED_MEMORY = """{"__version":"0.0.0","__root":{"__serializer":true,"__class":"UnconstrainedMemory","__value":{"messages":[{"__serializer":true,"__class":"UserMessage","__value":{"id":null,"meta":{"createdAt":{"__serializer":true,"__class":"datetime","__value":"2025-10-18T19:38:37.859543+00:00"}},"role":"user","content":[{"type":"text","text":"Hello!"}]}},{"__serializer":true,"__class":"AssistantMessage","__value":{"id":null,"meta":{"createdAt":{"__serializer":true,"__class":"datetime","__value":"2025-10-18T19:38:37.859665+00:00"}},"role":"assistant","content":[{"type":"text","text":"Hello, how can I help you?"}]}}]}}}"""


def main() -> None:
    memory = UnconstrainedMemory.from_serialized(SERIALIZED_MEMORY, extra_classes=[UserMessage, AssistantMessage])
    print([message.text for message in memory.messages])


if __name__ == "__main__":
    main()
