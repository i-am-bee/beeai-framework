# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

"""Compatibility shims for acp-sdk, whose final release (1.0.3) predates the uvicorn versions this package uses."""

try:
    import uvicorn.config as _uvicorn_config
except ModuleNotFoundError:  # pragma: no cover - uvicorn is part of the [acp] extra
    _uvicorn_config = None

# acp-sdk annotates `ACPServer.run`/`serve` with `uvicorn.config.LoopSetupType`, which uvicorn 0.36 renamed to
# `LoopFactoryType` (same values). The annotation is evaluated when `acp_sdk.server` is imported.
if (
    _uvicorn_config is not None
    and not hasattr(_uvicorn_config, "LoopSetupType")
    and hasattr(_uvicorn_config, "LoopFactoryType")
):
    _uvicorn_config.LoopSetupType = _uvicorn_config.LoopFactoryType  # type: ignore[attr-defined]
