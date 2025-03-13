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

import asyncio
import sys
import traceback

from beeai_framework.errors import FrameworkError
from beeai_framework.tools.python import PythonTool
from beeai_framework.tools.storage import Input, LocalPythonStorage


async def main() -> None:
    p = PythonTool(
        {
            "codeInterpreter": {"url": "http://127.0.0.1:50081"},
            "storage": LocalPythonStorage(Input("./localTmp", "./pythonTmp", [])),
        }
    )
    result = await p.run(
        {
            "language": "python",
            "code": "print(str(1+1))",
            "inputFiles": ["a", "c"],
        }
    )
    print(result)

    result = await p.run(
        {
            "language": "python",
            "code": "print(str(3+1))",
            "inputFiles": ["a", "c"],
        }
    )
    print(result)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
