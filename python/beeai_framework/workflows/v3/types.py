# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Coroutine
from typing import Any

from beeai_framework.backend.message import AnyMessage

ControllerFunction = Callable[[list[AnyMessage], dict[str, Any]], Any]
AsyncStepFunction = Callable[[list[AnyMessage], dict[str, Any]], Coroutine[Any, Any, Any]]
