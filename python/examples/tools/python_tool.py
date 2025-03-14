import asyncio
import sys
import traceback

from beeai_framework.adapters.ollama.backend.chat import OllamaChatModel
from beeai_framework.agents.react.agent import ReActAgent
from beeai_framework.errors import FrameworkError
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.python import PythonTool
from beeai_framework.tools.storage import Input, LocalPythonStorage


async def main() -> None:
    llm = OllamaChatModel("llama3.1")
    p = PythonTool(
        {
            "codeInterpreter": {"url": "http://127.0.0.1:50081"},
            "storage": LocalPythonStorage(Input("./localTmp", "./pythonTmp", [])),
        }
    )
    agent = ReActAgent(llm=llm, tools=[p], memory=UnconstrainedMemory())
    result = await agent.run("What's 1 plus 1?")
    print(result.result.text)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
