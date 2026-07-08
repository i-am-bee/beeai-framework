# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

"""Pytest configuration for asyncio testing."""

from pytest import Parser


def pytest_addoption(parser: Parser) -> None:
    """Add pytest command line options."""
    parser.addini("asyncio_mode", "default mode for async fixtures", default="strict")
