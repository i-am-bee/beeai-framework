# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Coroutine
from typing import Any

AsyncFunc = Callable[..., Coroutine[Any, Any, Any]]
