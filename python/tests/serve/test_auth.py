# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import pytest

from beeai_framework.serve.utils import is_api_key_valid


@pytest.mark.unit
class TestIsApiKeyValid:
    def test_no_configured_key_disables_auth(self) -> None:
        assert is_api_key_valid(None, None) is True
        assert is_api_key_valid(None, "anything") is True

    def test_missing_received_key_is_rejected(self) -> None:
        assert is_api_key_valid("secret", None) is False

    def test_matching_key(self) -> None:
        assert is_api_key_valid("secret", "secret") is True

    def test_mismatched_key(self) -> None:
        assert is_api_key_valid("secret", "wrong") is False

    def test_bearer_prefix_is_stripped_when_requested(self) -> None:
        assert is_api_key_valid("secret", "Bearer secret", strip_bearer_prefix=True) is True

    def test_bearer_prefix_is_not_stripped_by_default(self) -> None:
        assert is_api_key_valid("secret", "Bearer secret") is False

    @pytest.mark.parametrize("header", ["bearer secret", "BEARER secret", "Bearer    secret"])
    def test_bearer_prefix_is_case_insensitive_and_tolerates_whitespace(self, header: str) -> None:
        assert is_api_key_valid("secret", header, strip_bearer_prefix=True) is True

    def test_only_the_leading_prefix_is_stripped(self) -> None:
        """Regression: an unanchored strip removed "Bearer " from anywhere in the key.

        A configured key that itself contains "Bearer " was wrongly rejected even when
        the caller sent a correctly formed header.
        """
        key = "sk-Bearer team1"
        assert is_api_key_valid(key, f"Bearer {key}", strip_bearer_prefix=True) is True

    def test_key_without_bearer_header_still_matches(self) -> None:
        assert is_api_key_valid("secret", "secret", strip_bearer_prefix=True) is True

    def test_non_ascii_key(self) -> None:
        assert is_api_key_valid("kľúč-🔑", "kľúč-🔑") is True
        assert is_api_key_valid("kľúč-🔑", "kluc-🔑") is False

    def test_empty_received_key_is_rejected(self) -> None:
        assert is_api_key_valid("secret", "") is False
