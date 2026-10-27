# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import warnings
from typing import Any

AGENTSTACK_DEPRECATION_MESSAGE = (
    "AgentStack is no longer maintained. The AgentStack integration (AgentStackAgent, AgentStackServer, "
    "AgentStackChatModel, the 'agentstack' provider and the [agentstack] / [beeai-platform] extras) "
    "is deprecated and will be removed in a future release. "
    "For agents hosted on an A2A-compatible server, use beeai_framework.adapters.a2a (A2AAgent / A2AServer) instead."
)


def _user_stacklevel() -> int:
    """Return a `stacklevel` that points the warning at the first frame outside the framework.

    The package is imported either directly by the user, through the deprecated `beeai_platform` alias,
    or lazily by `ChatModel.from_name("agentstack:...")` and similar helpers. Skipping the framework and
    import machinery frames makes the warning point at the user's code in all of these cases.
    """
    framework_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + os.sep
    frame = sys._getframe(1)  # the module body of this package
    level = 1
    while frame.f_back is not None:
        frame = frame.f_back
        filename = frame.f_code.co_filename
        # `warnings.warn` does not count the frozen importlib bootstrap frames, so neither do we.
        if "importlib" in filename and "_bootstrap" in filename:
            continue
        level += 1
        if not filename.startswith(framework_root) and "importlib" not in filename:
            break
    return level


warnings.warn(AGENTSTACK_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=_user_stacklevel())


def __getattr__(name: str) -> Any:
    if name in {"AgentStackAgent", "AgentStackAgentErrorEvent", "AgentStackAgentOutput", "AgentStackAgentUpdateEvent"}:
        import beeai_framework.adapters.agentstack.agents.agent as agent

        warnings.warn(
            f"Please import {name} from beeai_framework.adapters.agentstack.agents.agent instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        return getattr(agent, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
