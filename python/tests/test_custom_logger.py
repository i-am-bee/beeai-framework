# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import logging

import pytest

from beeai_framework.logger import Logger

"""
Unit Tests
"""


@pytest.mark.unit
def test_redefine_logging_methods() -> None:
    logger = Logger("app", level=logging.DEBUG)
    logger.add_logging_level("TEST1", 1, "test")  # adds test log level
    logger.add_logging_level("TEST2", 2, "test")  # does not redefine test log level
    logger.add_logging_level("INFO", logging.INFO)  # does not redefine info log level
    assert callable(logger.test)  # type: ignore
