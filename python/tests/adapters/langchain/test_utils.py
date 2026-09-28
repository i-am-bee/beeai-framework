# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import pytest

from beeai_framework.adapters.langchain.backend._utils import (
    to_beeai_message_content,
    to_beeai_messages,
    to_lc_message_content,
    to_lc_messages,
)
from beeai_framework.backend.message import MessageImageContent, UserMessage


@pytest.mark.unit
@pytest.mark.parametrize(
    ("lc_content", "beeai_url"),
    [
        ({"type": "image", "source_type": "url", "url": "https://example.com/cat.png"}, "https://example.com/cat.png"),
        (
            {"type": "image", "source_type": "base64", "mime_type": "image/png", "data": "AQID"},
            "data:image/png;base64,AQID",
        ),
        ({"type": "image", "mime_type": "image/png", "base64": "AQID"}, "data:image/png;base64,AQID"),
    ],
)
def test_langchain_image_to_beeai(lc_content: dict[str, str], beeai_url: str) -> None:
    result = to_beeai_message_content(lc_content)

    assert isinstance(result, MessageImageContent)
    assert result.image_url["url"] == beeai_url


@pytest.mark.unit
def test_langchain_base64_image_without_mime_type_is_unsupported() -> None:
    assert to_beeai_message_content({"type": "image", "base64": "AQID"}) is None


@pytest.mark.unit
@pytest.mark.parametrize(
    ("beeai_url", "lc_content"),
    [
        (
            "https://example.com/cat.png",
            {"type": "image", "source_type": "url", "url": "https://example.com/cat.png"},
        ),
        (
            "data:image/png;base64,AQID",
            {"type": "image", "source_type": "base64", "mime_type": "image/png", "data": "AQID"},
        ),
    ],
)
def test_beeai_image_to_langchain(beeai_url: str, lc_content: dict[str, str]) -> None:
    image = UserMessage.from_image(beeai_url).content[0]

    assert to_lc_message_content(image) == lc_content


@pytest.mark.unit
@pytest.mark.parametrize("url", ["https://example.com/cat.png", "data:image/png;base64,AQID"])
def test_image_message_round_trip(url: str) -> None:
    original = UserMessage.from_image(url)

    result = to_beeai_messages(to_lc_messages([original]))

    assert isinstance(result[0].content[0], MessageImageContent)
    assert result[0].content[0].image_url["url"] == url
