import asyncio
import sys
import traceback

from beeai_framework.adapters.ollama.backend.chat import OllamaChatModel
from beeai_framework.agents.react.agent import ReActAgent
from beeai_framework.errors import FrameworkError
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.code import LocalPythonStorage, PythonTool


async def main() -> None:
    llm = OllamaChatModel("llama3.1")
    storage = LocalPythonStorage(local_working_dir="./localTmp", interpreter_working_dir="./pythonTmp")
    python_tool = PythonTool(
        code_interpreter_url="http://127.0.0.1:50081",
        storage=storage,
    )
    agent = ReActAgent(llm=llm, tools=[python_tool], memory=UnconstrainedMemory())
    result = await agent.run("What's 1 plus 1?")
    print(result.result.text)

    # delete temp dirs if empty
    storage.clean_up()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
