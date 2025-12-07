# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated

from typing_extensions import TypedDict

try:
    import agentstack_sdk.a2a.extensions as agentstack_extensions
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [agentstack] not found.\nRun 'pip install \"beeai-framework[agentstack]\"' to install."
    ) from e


class BaseAgentStackExtensions(TypedDict, total=True):
    form_request: Annotated[
        agentstack_extensions.FormRequestExtensionServer,
        agentstack_extensions.FormRequestExtensionSpec(),
    ]
    trajectory: Annotated[
        agentstack_extensions.TrajectoryExtensionServer, agentstack_extensions.TrajectoryExtensionSpec()
    ]
