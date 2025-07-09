# Copyright 2025 © BeeAI a Series of LF Projects, LLC
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

from abc import ABC, abstractmethod
from typing import Generic

from pydantic import BaseModel
from typing_extensions import TypeVar

from beeai_framework.backend.message import UserMessage
from beeai_framework.context import Run
from beeai_framework.utils import AbortSignal
from beeai_framework.utils.models import ModelLike


class RunnableConfig(BaseModel):
    """Configuration and runtime metadata for a Runnable."""

    """The runnable's abort signal data"""
    signal: AbortSignal | None = None


Input = TypeVar("Input", bound=str | UserMessage | ModelLike)
Output = TypeVar("Output", bound=str | ModelLike)
Config = TypeVar("Config", bound=RunnableConfig)


class Runnable(Generic[Input, Output, Config], ABC):
    """A unit of work that can be invoked using a stable interface."""

    @abstractmethod
    def run(self, input: Input, config: Config | None = None) -> Run[Output]: ...
