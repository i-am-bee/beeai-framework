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

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Concatenate, ParamSpec, TypeVar

from requests import Request

P = ParamSpec("P")
R = TypeVar("R", covariant=True)


def request_wrapper(f: Callable[Concatenate[Any, Request, P], R]) -> Callable[Concatenate[Any, P], R]:
    def inner(self: Any, *args: P.args, **kwargs: P.kwargs) -> R:
        return f(self, Request(), *args, **kwargs)

    return inner  # type: ignore


class Thing:
    @request_wrapper
    def takes_int_str(self, request: Request, x: int, y: str) -> int:
        print(request)
        return x + 7


thing = Thing()
thing.takes_int_str(1, "a")
