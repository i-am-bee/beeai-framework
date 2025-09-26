# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import sys
import warnings

import beeai_framework.agents.requirement.prompts as _new_module

warnings.warn(
    "beeai_framework.agents.experimental.prompts is deprecated and will be removed in a future release. "
    "Please use beeai_framework.agents.requirement.prompts instead.",
    DeprecationWarning,
    stacklevel=2,
)

sys.modules[__name__] = _new_module
