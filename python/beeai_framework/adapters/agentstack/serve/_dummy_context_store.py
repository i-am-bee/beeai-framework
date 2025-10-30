# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncIterator

try:
    import a2a.types as a2a_types
    import agentstack_sdk.server.store.context_store as agentstack_context_store
    from agentstack_sdk.server.dependencies import Dependency

except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [agentstack] not found.\nRun 'pip install \"beeai-framework[agentstack]\"' to install."
    ) from e


class DummyContextStoreInstance(agentstack_context_store.ContextStoreInstance):
    async def load_history(self) -> AsyncIterator[a2a_types.Message | a2a_types.Artifact]:
        if False:
            yield

    async def store(self, data: a2a_types.Message | a2a_types.Artifact) -> None:
        pass


class DummyContextStore(agentstack_context_store.ContextStore):
    _cs = DummyContextStoreInstance()

    async def create(
        self,
        context_id: str,
        initialized_dependencies: list[Dependency],  # type: ignore[type-arg]
    ) -> agentstack_context_store.ContextStoreInstance:
        return self._cs
