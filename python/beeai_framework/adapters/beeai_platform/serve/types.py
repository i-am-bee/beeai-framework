# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated, TypedDict

try:
    import beeai_sdk.a2a.extensions as beeai_extensions
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [beeai-platform] not found.\nRun 'pip install \"beeai-framework[beeai-platform]\"' to install."
    ) from e


class BaseBeeAIPlatformExtensions(TypedDict, total=True):
    form: Annotated[
        beeai_extensions.FormExtensionServer,
        beeai_extensions.FormExtensionSpec(params=None),
    ]
    trajectory: Annotated[beeai_extensions.TrajectoryExtensionServer, beeai_extensions.TrajectoryExtensionSpec()]
