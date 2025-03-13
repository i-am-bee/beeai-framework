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

import os
from unittest.mock import patch

import pytest

from beeai_framework.adapters.litellm import utils


class TestParseExtraHeaders:
    """
    Test cases for the parse_extra_headers function.
    This is used by adapters to send extra headers to the model.
    This can be useful for proxy servers and gateways
    """

    @pytest.mark.unit
    def test_settings_headers_dict(self) -> None:
        """

        Extra headers are provided in settings (dict).
        Typically when client app passes setting on ChatModel constructor
        """
        settings_headers = {"header1": "value1", "header2": "value2"}
        result = utils.parse_extra_headers(settings_headers, None)
        assert result == settings_headers

    @pytest.mark.unit
    def test_no_headers(self) -> None:
        """
        Test when no headers are provided, it should return {}
        The most regular case - no extra headers needed
        """
        result = utils.parse_extra_headers(None, None)
        assert result == {}

    @pytest.mark.unit
    def test_env_headers(self) -> None:
        """
        Test when extra headers are provided as an environment variable.
        Simple usage, avoiding any code change
        """
        with patch.dict(os.environ, {"LITELLM_EXTRA_HEADERS": "header1=value1, header2=value2"}):
            result = utils.parse_extra_headers(None, os.environ.get("LITELLM_EXTRA_HEADERS"))
            assert result == {"header1": "value1", "header2": "value2"}

    @pytest.mark.unit
    def test_settings_override_env(self) -> None:
        """
        Test that existing settings override the environment variables.
        """
        settings_headers = {"header1": "new_value1", "header3": "value3"}
        with patch.dict(os.environ, {"ENV_VAR": "header1=value1, header2=value2"}):
            result = utils.parse_extra_headers(settings_headers, os.environ.get("ENV_VAR"))
            assert result == {"header1": "new_value1", "header3": "value3"}

    @pytest.mark.unit
    def test_env_headers_with_spaces(self) -> None:
        """Test when environment variable has extra spaces."""
        with patch.dict(os.environ, {"ENV_VAR": " header1 = value1 , header2= value2 "}):
            result = utils.parse_extra_headers(None, os.environ.get("ENV_VAR"))
            assert result == {"header1": "value1", "header2": "value2"}

    @pytest.mark.unit
    def test_invalid_headers_string_format(self) -> None:
        """
        Test when a malformed header string is passed.
        Expected behaviour is that only correct key=value pairs are parsed.
        """
        with patch.dict(os.environ, {"ENV_VAR": "header1=value1,header2value2"}):
            result = utils.parse_extra_headers(None, os.environ.get("ENV_VAR"))
            assert result == {"header1": "value1"}

    @pytest.mark.unit
    def test_empty_header_string(self) -> None:
        """Test when environment variable is empty or whitespace."""

        with patch.dict(os.environ, {"ENV_VAR": ""}):
            result = utils.parse_extra_headers(None, os.environ.get("ENV_VAR"))
            assert result == {}

        with patch.dict(os.environ, {"ENV_VAR": "    "}):
            result = utils.parse_extra_headers(None, os.environ.get("ENV_VAR"))
            assert result == {}

    @pytest.mark.unit
    def test_settings_headers_none(self) -> None:
        """Test when settings headers are explicitly set to None."""
        result = utils.parse_extra_headers(None, None)
        assert result == {}
