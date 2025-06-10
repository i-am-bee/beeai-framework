# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import re


def update(path: str) -> None:
    with open(path, encoding="utf-8") as file:
        content = file.read()

    pattern1 = r"(\[.*?\]\()(?!https?:\/\/)(.*?)\)"
    replacement1 = "\\1https://github.com/i-am-bee/beeai-framework/tree/main\\2)"
    content = re.sub(pattern1, replacement1, content, flags=re.MULTILINE)

    with open(path, "w", encoding="utf-8") as file:
        file.write(content)
