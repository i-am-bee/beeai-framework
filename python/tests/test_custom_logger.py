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

import logging

import pytest

from beeai_framework.backend import Role
from beeai_framework.logger import Logger
from beeai_framework.utils import MessageEvent

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


@pytest.mark.unit
def test_log_events() -> None:
    logger = Logger("app")
    event = MessageEvent(source=Role.USER, message="Test")
    logger.log_message_events(event)
    logger.info("Test", extra={"is_event_message": False})
