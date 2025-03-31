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

from typing import Any

from beeai_framework.errors import FrameworkError
from beeai_framework.serializer.serializable import Serializable


class RetryCounter(Serializable):
    def __init__(self, error_type: type[BaseException], max_retries: int = 0) -> None:
        if not issubclass(error_type, FrameworkError):
            raise ValueError("error_type must be a subclass of FrameworkError")

        self._max_retries = max_retries
        self.error_type = error_type
        self.remaining = max_retries

        self._error_class: type[BaseException] = error_type  # TODO: FrameworkError
        self._lastError: BaseException | None = None
        self._finalError: BaseException | None = None

    def use(self, error: BaseException) -> None:
        if self._finalError:
            raise self._finalError

        self._lastError = error or self._lastError
        self.remaining -= 1

        # TODO: ifFatal, isRetryable etc
        if self.remaining < 0:
            self._finalError = self._error_class(  # type: ignore
                f"Maximal amount of global retries ({self._max_retries}) has been reached.", cause=self._lastError
            )
            raise self._finalError

    async def create_snapshot(self) -> dict[str, Any]:
        return {
            "remaining": self.remaining,
            "max_retries": self._max_retries,
            "last_error": self._lastError,
            "final_error": self._finalError,
            "error_class": self._error_class,
        }

    async def load_snapshot(self, snapshot: dict[str, Any]) -> None:
        self.remaining = snapshot["remaining"]
        self._max_retries = snapshot["max_retries"]
        self._lastError = snapshot["last_error"]
        self._finalError = snapshot["final_error"]
        self._error_class = snapshot["error_class"]
