# Copyright 2025 IBM Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import uuid
import weakref
from asyncio import AbstractEventLoop
from typing import Any

import litellm

__all__ = ["_patch_litellm_cache"]


class EventLoopAwareInMemoryCache(litellm.InMemoryCache):  # type: ignore
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._clients: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()  # type: ignore

    def get_cache(self, key: str, **kwargs: Any) -> Any | None:
        new_key = self._create_cache_key(key)
        return super().get_cache(new_key)

    def set_cache(self, key: str, value: Any, **kwargs: Any) -> None:
        new_key = self._create_cache_key(key)
        super().set_cache(new_key, value, **kwargs)

    def _create_cache_key(self, key: str) -> str:
        loop: AbstractEventLoop = asyncio.get_running_loop()
        if loop not in self._clients:
            self._clients[loop] = uuid.uuid4()

        loop_id = self._clients[loop]
        return f"{key}-{loop_id}"


def _patch_litellm_cache() -> None:
    """Fixes https://github.com/BerriAI/litellm/issues/7667"""
    litellm.in_memory_llm_clients_cache = EventLoopAwareInMemoryCache()
