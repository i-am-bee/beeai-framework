# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import sys
import warnings

import beeai_framework.agents.requirement.agent as _new_module

warnings.warn(
    "beeai_framework.agents.experimental.agent is deprecated and will be removed in a future release. "
    "Please use beeai_framework.agents.requirement.agent instead.",
    DeprecationWarning,
    stacklevel=2,
)

sys.modules[__name__] = _new_module
