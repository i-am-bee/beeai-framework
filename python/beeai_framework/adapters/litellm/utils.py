# Copyright 2025 IBM Corp.
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

from typing import Any

from beeai_framework.logger import Logger

logger = Logger(__name__)


def parse_extra_headers(
    settings_headers: dict[str, Any] | None = None,
    env_headers: str | None = None,
) -> dict[str, str]:
    """
    Parses extra headers from settings and/or environment variables.

    Args:
        settings_headers: Extra headers provided in settings.
        env_headers: Extra headers from the environment variable.

    Returns:
        A dictionary of extra headers or {} if no headers are found.

    """
    extra_headers: dict[str, str] = {}

    # Priority 2: Environment variable (provider-specific)
    if env_headers:
        extra_headers = _parse_header_string(env_headers)

    # Priority 1: Settings (highest priority)
    if settings_headers is None:
        pass
    elif isinstance(settings_headers, dict):
        extra_headers = settings_headers
    else:
        # Should be unreachable. protect against not passing dict/None
        raise ValueError(
            f"Invalid settings_headers format. Expected a dictionary or None, received {type(settings_headers)}"
        )

    return extra_headers


def _parse_header_string(header_string: str) -> dict[str, str]:
    """Parses a comma-separated string of headers into a dictionary."""
    headers: dict[str, str] = {}
    if not header_string:
        return headers
    header_string = header_string.strip()
    for pair in header_string.split(","):
        pair = pair.strip()
        if "=" in pair:
            key, value = pair.split("=", 1)
            headers[key.strip()] = value.strip()
        else:
            logger.warning(f"Malformed header string detected. Will ignore it: {pair}")
    return headers
