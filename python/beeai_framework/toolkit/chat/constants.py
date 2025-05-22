# Copyright 2025 © BeeAI a Series of LF Projects, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Constants for the chat model plugins."""

PROMPT_SYSTEM = "You are a friendly agent."

USER_PROMPT_VARIABLE = "user"
ASSISTANT_PROMPT_VARIABLE = "assistant"

PROMPT_SUFFIX = "{{user}}"
PROMPT_PREFIX = ""

USER_EXAMPLE = PROMPT_SUFFIX
ASSISTANT_EXAMPLE = "{{assistant}}"

PROMPT_STOP = ["\nInput:"]

INPUT = "input"

"""Chat Model options."""
MODEL_ID = "model_id"

FINISH_REASON = "finish_reason"
PROMPT_TOKENS = "prompt_tokens"
COMPLETION_TOKENS = "completion_tokens"
TOTAL_TOKENS = "total_tokens"
