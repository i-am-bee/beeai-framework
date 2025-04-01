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
import sys
from logging import Formatter
from typing import TYPE_CHECKING, Any

from beeai_framework.errors import FrameworkError
from beeai_framework.utils.config import CONFIG


class LoggerError(FrameworkError):
    """Raised for errors caused by logging."""

    def __init__(
        self,
        message: str = "Logger error",
        *,
        cause: Exception | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, is_fatal=True, is_retryable=False, cause=cause, context=context)


class LoggerFormatter(Formatter):
    def format(self, record: logging.LogRecord) -> str:
        if hasattr(record, "is_event_message") and record.is_event_message:
            return logging.Formatter(
                "{asctime} | {levelname:<8s} |{message}",
                style="{",
                datefmt="%Y-%m-%d %H:%M:%S",
            ).format(record)
        else:
            return logging.Formatter(
                "{asctime} | {levelname:<8s} | {name}:{funcName}:{lineno} - {message}",
                style="{",
                datefmt="%Y-%m-%d %H:%M:%S",
            ).format(record)


class Logger(logging.Logger):
    if TYPE_CHECKING:
        trace = logging.Logger.debug

    def __init__(self, name: str, level: int | str = CONFIG.log_level) -> None:
        self.add_logging_level("TRACE", logging.DEBUG - 5)

        super().__init__(name, level)

        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setFormatter(LoggerFormatter())

        self.addHandler(console_handler)

    # https://stackoverflow.com/questions/2183233/how-to-add-a-custom-loglevel-to-pythons-logging-facility/35804945#35804945
    def add_logging_level(self, level_name: str, level_num: int, method_name: str | None = None) -> None:
        """
        Comprehensively adds a new logging level to the `logging` module and the
        currently configured logging class.

        `level_name` becomes an attribute of the `logging` module with the value
        `level_num`. `method_name` becomes a convenience method for both `logging`
        itself and the class returned by `logging.getLoggerClass()` (usually just
        `logging.Logger`). If `method_name` is not specified, `level_name.lower()` is
        used.

        To avoid accidental clobberings of existing attributes, this method will
        return without action if the level name is already an attribute of the
        `logging` module or if the method name is already present

        Example
        -------
        >>> add_logging_level('TRACE', logging.DEBUG - 5)
        >>> logging.getLogger(__name__).setLevel("TRACE")
        >>> logging.getLogger(__name__).trace('that worked')
        >>> logging.trace('so did this')
        >>> logging.TRACE
        5

        """
        if not method_name:
            method_name = level_name.lower()

        if hasattr(logging, level_name):
            # already defined in logging module
            return
        if hasattr(logging, method_name):
            # already defined in logging module
            return
        if hasattr(logging.getLoggerClass(), method_name):  # pragma: no cover
            # already defined in logger class
            return

        # This method was inspired by the answers to Stack Overflow post
        # http://stackoverflow.com/q/2183233/2988730, especially
        # http://stackoverflow.com/a/13638084/2988730
        def log_for_level(self: logging.Logger, message: str, *args: int, **kwargs: Any) -> None:  # pragma: no cover
            if self.isEnabledFor(level_num):
                self._log(level_num, message, args, stacklevel=2, **kwargs)

        def log_to_root(message: str, *args: int, **kwargs: Any) -> None:  # pragma: no cover
            logging.log(level_num, message, *args, **kwargs)

        logging.addLevelName(level_num, level_name)
        setattr(logging, level_name, level_num)
        setattr(logging.getLoggerClass(), method_name, log_for_level)
        setattr(logging, method_name, log_to_root)


__all__ = ["Logger", "LoggerError"]
