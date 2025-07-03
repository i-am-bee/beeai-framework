# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from beeai_framework.adapters.acp.serve.server import ACPServerConfig


class BeeAIPlatformServerConfig(ACPServerConfig):
    """Configuration for the Beeai Server."""

    self_registration: bool | None = True
