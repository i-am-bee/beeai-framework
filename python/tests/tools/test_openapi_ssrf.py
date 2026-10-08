# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import pytest

from beeai_framework.tools.errors import ToolError
from beeai_framework.tools.openapi import OpenAPITool

# ── _validate_path_safe tests ────────────────────────────────────────────────


@pytest.mark.unit
class TestValidatePathSafe:
    """Tests for the _validate_path_safe static method that blocks URL override via urljoin."""

    def test_absolute_http_url_rejected(self) -> None:
        with pytest.raises(ToolError, match="Absolute URLs are not allowed"):
            OpenAPITool._validate_path_safe("https://api.example.com", "http://169.254.169.254/latest/meta-data/")

    def test_absolute_https_url_rejected(self) -> None:
        with pytest.raises(ToolError, match="Absolute URLs are not allowed"):
            OpenAPITool._validate_path_safe("https://api.example.com", "https://attacker.com/evil")

    def test_protocol_relative_url_rejected(self) -> None:
        with pytest.raises(ToolError, match="Absolute URLs are not allowed"):
            OpenAPITool._validate_path_safe("https://api.example.com", "//169.254.169.254/latest/meta-data/")

    def test_scheme_like_path_rejected(self) -> None:
        with pytest.raises(ToolError, match="Path must be a relative path"):
            OpenAPITool._validate_path_safe("https://api.example.com", "ftp://internal/file")

    def test_normal_absolute_path_allowed(self) -> None:
        result = OpenAPITool._validate_path_safe("https://api.example.com/v1", "/users")
        assert result == "https://api.example.com/users"

    def test_normal_relative_path_allowed(self) -> None:
        result = OpenAPITool._validate_path_safe("https://api.example.com/v1/", "users")
        assert result == "https://api.example.com/v1/users"

    def test_path_with_params_allowed(self) -> None:
        result = OpenAPITool._validate_path_safe("https://api.example.com", "/users/{id}")
        assert result == "https://api.example.com/users/{id}"

    def test_empty_path_allowed(self) -> None:
        result = OpenAPITool._validate_path_safe("https://api.example.com/v1", "")
        assert result == "https://api.example.com/v1"

    def test_none_base_url_rejected(self) -> None:
        with pytest.raises(ToolError, match="Base URL is required"):
            OpenAPITool._validate_path_safe(None, "/users")
