import asyncio
import os
import sys
import tempfile
import traceback

from dotenv import load_dotenv

from beeai_framework.adapters.ollama.backend.chat import OllamaChatModel
from beeai_framework.agents.react.agent import ReActAgent
from beeai_framework.errors import FrameworkError
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.code import LocalPythonStorage, PythonTool

# Load environment variables
load_dotenv()


async def main() -> None:
    llm = OllamaChatModel("llama3.1")
    storage = LocalPythonStorage(
        local_working_dir=tempfile.mkdtemp("code_interpreter_source"),
        # CODE_INTERPRETER_TMPDIR should point to where code interpreter stores it's files
        interpreter_working_dir=os.getenv("CODE_INTERPRETER_TMPDIR", "./tmp/code_interpreter_target"),
    )
    python_tool = PythonTool(
        code_interpreter_url=os.getenv("CODE_INTERPRETER_URL", "http://127.0.0.1:50081"),
        storage=storage,
    )
    agent = ReActAgent(llm=llm, tools=[python_tool], memory=UnconstrainedMemory())
    result = await agent.run("Calculate 5036 * 12856 and save the result to answer.txt").on(
        "update", lambda data, event: print(f"Agent 🤖 ({data.update.key}) : ", data.update.parsed_value)
    )
    print(result.result.text)

    result = await agent.run("Read the content of answer.txt?").on(
        "update", lambda data, event: print(f"Agent 🤖 ({data.update.key}) : ", data.update.parsed_value)
    )
    print(result.result.text)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
