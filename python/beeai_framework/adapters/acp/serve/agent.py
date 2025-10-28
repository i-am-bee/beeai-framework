"""Module for the ACP server agent."""
# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import contextlib
from collections.abc import AsyncGenerator, Callable

from beeai_framework.adapters.acp.serve.io import ACPIOContext

try:
    import acp_sdk.models as acp_models
    import acp_sdk.server.context as acp_context
    import acp_sdk.server.types as acp_types
    from acp_sdk.server.agent import Agent as ACPBaseAgent
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [acp] not found.\nRun 'pip install \"beeai-framework[acp]\"' to install."
    ) from e


class ACPServerAgent(ACPBaseAgent):
    """A wrapper for a BeeAI agent to be used with the ACP server."""

    def __init__(
        self,
        fn: Callable[
            [list[acp_models.Message], acp_context.Context],
            AsyncGenerator[acp_types.RunYield, acp_types.RunYieldResume],
        ],
        name: str,
        description: str | None = None,
        metadata: acp_models.Metadata | None = None,
    ) -> None:
        super().__init__()
        self.fn = fn
        self._name = name
        self._description = description
        self._metadata = metadata

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description or ""

    @property
    def metadata(self) -> acp_models.Metadata:
        return self._metadata or acp_models.Metadata()

    async def run(
        self, input: list[acp_models.Message], context: acp_context.Context
    ) -> AsyncGenerator[acp_types.RunYield, acp_types.RunYieldResume]:
        """
        Execute the agent with the given input messages and context.

        Args:
            input (list[acp_models.Message]): A list of ACP messages to process.
            context (acp_context.Context): The ACP server context.
        Yields:
            RunYield values produced by the agent during execution.
        Receives:
            RunYieldResume values to continue execution after yielding.
        """
        with ACPIOContext(context), contextlib.suppress(StopAsyncIteration):
            gen = self.fn(input, context)
            value = None
            while True:
                value = yield await gen.asend(value)
