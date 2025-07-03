# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

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
