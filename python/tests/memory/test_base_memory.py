# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import pytest

from beeai_framework.backend.message import UserMessage
from beeai_framework.memory.unconstrained_memory import UnconstrainedMemory


@pytest.mark.asyncio
@pytest.mark.unit
async def test_delete_many_with_filtered_memory_iterator() -> None:
    memory = UnconstrainedMemory()
    keep = UserMessage("keep")
    await memory.add_many([UserMessage("old"), UserMessage("old"), keep])

    await memory.delete_many(message for message in memory if message.text == "old")

    assert memory.messages == [keep]
