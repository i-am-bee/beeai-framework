import asyncio
import sys

from dotenv import load_dotenv

from beeai_framework.agents.react import ReActAgent
from beeai_framework.backend.chat import ChatModel
from beeai_framework.memory.unconstrained_memory import UnconstrainedMemory

LLMS = {
    "ollama": "ollama:llama3.1",
    "watsonx": "watsonx:ibm/granite-3-8b-instruct",
}

HELP = """
Usage:
  examples.beeai_framework.llms <ollama|openai|watsonx>
Arguments
  `ollama` - requires local ollama service running (i.e., http://127.0.0.1:11434)
  `watsonx` - requires environment variable
      - WATSONX_URL - base URL of your WatsonX instance
    and one of the following
      - WATSONX_API_KEY: API key
      - WATSONX_TOKEN: auth token
"""


async def main(name: str) -> None:
    chat_model = ChatModel.from_name(name)
    agent = ReActAgent(llm=chat_model, tools=[], memory=UnconstrainedMemory())

    result = await agent.run("What is the smallest of the Cabo Verde islands?")

    print("answer:", result.result.text)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "--help":
        print(HELP)
    else:
        load_dotenv()
        model = LLMS.get(sys.argv[1])
        if model:
            asyncio.run(main(model))
        else:
            print(f"Unknown provider: {sys.argv[1]}\n{HELP}")
