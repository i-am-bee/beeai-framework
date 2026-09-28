# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from beeai_framework.cache.base import BaseCache
from beeai_framework.cache.sliding_cache import SlidingCache
from beeai_framework.cache.unconstrained_cache import UnconstrainedCache
from beeai_framework.context import RunContext
from beeai_framework.tools import StringToolOutput, ToolOutput
from beeai_framework.tools.code import ShellTool
from beeai_framework.tools.filesystem import FileEditTool, FileReadTool, GlobTool, GrepTool
from beeai_framework.tools.tool import AnyTool

pytestmark = [
    pytest.mark.unit,
    pytest.mark.asyncio,
    pytest.mark.parametrize("tool_type", [FileReadTool, FileEditTool, GlobTool, GrepTool, ShellTool]),
]


async def test_clone_preserves_middlewares(tool_type: type[AnyTool], tmp_path: Path) -> None:
    path = tmp_path / "input.txt"
    path.write_text("hello\n")
    inputs: dict[type[AnyTool], dict[str, Any]] = {
        FileReadTool: {"path": str(path)},
        FileEditTool: {"path": str(path), "mode": "overwrite", "content": "hello\n"},
        GlobTool: {"root": str(tmp_path), "pattern": "*.txt"},
        GrepTool: {"root": str(tmp_path), "pattern": "hello"},
        ShellTool: {"command": [sys.executable, "-c", "print('hello')"]},
    }
    calls: list[str] = []

    def first(context: RunContext) -> None:
        calls.append("first")

    def second(context: RunContext) -> None:
        calls.append("second")

    original = tool_type()
    original.middlewares.extend([first, second])
    cloned = await original.clone()
    recloned = await cloned.clone()

    for instance in (original, cloned, recloned):
        calls.clear()
        await instance.run(inputs[tool_type])
        assert calls == ["first", "second"]
        assert instance.cache.enabled is False

    cloned.middlewares.clear()
    assert original.middlewares == [first, second]
    assert recloned.middlewares == [first, second]


@pytest.mark.parametrize(
    "cache_factory",
    [UnconstrainedCache, lambda: SlidingCache(size=4), lambda: SlidingCache(size=4, ttl=60)],
    ids=["unconstrained", "sliding", "sliding-ttl"],
)
async def test_clone_isolates_cache(
    tool_type: type[AnyTool], cache_factory: Callable[[], BaseCache[ToolOutput]]
) -> None:
    original = tool_type(options={"cache": cache_factory()})
    value = StringToolOutput("cached result")
    await original.cache.set("shared", value)

    cloned = await original.clone()
    assert await cloned.cache.get("shared") is value
    await cloned.cache.delete("shared")
    assert await original.cache.get("shared") is value

    await cloned.cache.set("clone-only", value)
    assert not await original.cache.has("clone-only")
    recloned = await cloned.clone()
    assert await recloned.cache.get("clone-only") is value
    assert not await recloned.cache.has("shared")

    await original.cache.set("original-only", value)
    assert not await cloned.cache.has("original-only")
    await original.clear_cache()
    assert await cloned.cache.get("clone-only") is value
    await cloned.clear_cache()
    assert await recloned.cache.get("clone-only") is value
