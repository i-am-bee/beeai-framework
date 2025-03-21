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


from typing import TypeVar

from beeai_framework.cache.base import BaseCache

T = TypeVar("T")


class NullCache(BaseCache[T]):
    def __init__(self) -> None:
        super().__init__()
        self._enabled: bool = False

    async def size(self) -> int:
        return 0

    async def set(self, _key: str, _value: T) -> None:
        pass

    async def get(self, key: str) -> T | None:
        return None

    async def has(self, key: str) -> bool:
        return False

    async def delete(self, key: str) -> bool:
        return True

    async def clear(self) -> None:
        pass
