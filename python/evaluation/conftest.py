# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from typing import Any

from dotenv import load_dotenv


def pytest_configure(config: Any) -> None:
    load_dotenv()
    os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"
    os.environ["DEEPTEAM_TELEMETRY_OPT_OUT"] = "YES"
    os.environ["TELEMETRY_OPT_OUT"] = "YES"
    os.environ["DEEPEVAL_UPDATE_WARNING_OPT_OUT"] = "YES"
    os.environ["DEEPEVAL_UPDATE_WARNING_OPT_IN"] = "NO"
    os.environ["ERROR_REPORTING"] = "NO"

    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.CRITICAL)
    logging.disable(logging.DEBUG)
