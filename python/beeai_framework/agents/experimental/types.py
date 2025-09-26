# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import sys
import warnings

import beeai_framework.agents.requirement.types as _new_module

warnings.warn(
    "beeai_framework.agents.experimental.types is deprecated and will be removed in a future release. "
    "Please use beeai_framework.agents.requirement.types instead.",
    DeprecationWarning,
    stacklevel=2,
)

sys.modules[__name__] = _new_module
