# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, ConfigDict

from beeai_framework.backend.utils import parse_broken_json
from beeai_framework.utils.strings import find_first_pair


class GroqChatModelError(BaseModel):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    message: str
    type: str
    code: str
    failed_generation: str

    @staticmethod
    def parse_from_litellm_error(error: str) -> "GroqChatModelError":
        match = find_first_pair(error, ("{", "}"))
        if not match:
            raise ValueError("Unable to parse the provided error message.")

        result = parse_broken_json(match.outer, {})
        return GroqChatModelError.model_validate(result.get("error", {}))
