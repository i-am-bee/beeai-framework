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

from typing import Any, Self

from typing_extensions import Unpack, override

from beeai_framework.adapters.acp.serve.server import ACPServer, ACPServerMetadata, AnyAgentLike
from beeai_framework.adapters.beeai_platform.serve.agent import (
    BeeAIPlatformServerConfig,
)
from beeai_framework.utils.models import ModelLike, to_model


class BeeAIPlatformServerMetadata(ACPServerMetadata, total=False):
    # ui is native parameter for beeai platform
    ui: dict[str, Any]


class BeeAIPlatformServer(ACPServer):
    def __init__(self, *, config: ModelLike[BeeAIPlatformServerConfig] | None = None) -> None:
        super().__init__(config=to_model(BeeAIPlatformServerConfig, config or {"self_registration": True}))

    @override
    def register(self, input: AnyAgentLike, **metadata: Unpack[BeeAIPlatformServerMetadata]) -> Self:
        copy = metadata.copy()
        ui = copy.pop("ui", None)
        if ui:
            if copy.get("extra"):
                copy["extra"] = {**copy["extra"], "ui": ui}
            else:
                copy["extra"] = {"ui": ui}
        super().register(input, **copy)  # type: ignore
        return self
