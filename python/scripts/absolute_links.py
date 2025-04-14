# Copyright 2025 © BeeAI a Series of LF Projects, LLC
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

import re


def update(path: str) -> None:
    with open(path, encoding="utf-8") as file:
        content = file.read()

    pattern1 = r"(\[.*?\]\()(?!https?:\/\/)(.*?)\)"
    replacement1 = "\\1https://github.com/i-am-bee/beeai-framework/tree/main\\2)"
    content = re.sub(pattern1, replacement1, content, flags=re.MULTILINE)

    with open(path, "w", encoding="utf-8") as file:
        file.write(content)
