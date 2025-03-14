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
