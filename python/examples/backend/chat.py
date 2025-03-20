import asyncio
import sys
import traceback

from beeai_framework import UserMessage
from beeai_framework.adapters.ollama.backend.chat import OllamaChatModel
from beeai_framework.errors import FrameworkError
from examples.helpers.io import ConsoleReader


async def main() -> None:
    llm = OllamaChatModel("llama3.1")

    #  Optionally one may set llm parameters
    llm.parameters.max_tokens = 10000  # high number yields longer potential output
    llm.parameters.top_p = 0  # higher number yields more complex vocabulary, recommend only changing p or k
    llm.parameters.frequency_penalty = 0  # higher number yields reduction in word reptition
    llm.parameters.temperature = 0  # higher number yields greater randomness and variation
    llm.parameters.top_k = 0  # higher number yields more variance, recommend only changing p or k
    llm.parameters.n = 1  # higher number yields more choices
    llm.parameters.presence_penalty = 0  # higher number yields reduction in repetition of words
    llm.parameters.seed = 10  # can help produce similar responses if prompt and seed are always the same
    llm.parameters.stop_sequences = ["q", "quit", "ahhhhhhhhh"]  # stops the model on input of any of these strings
    llm.parameters.stream = False  # determines whether or not to use streaming to receive incremental data

    reader = ConsoleReader()

    for prompt in reader:
        response = await llm.create(messages=[UserMessage(prompt)])
        reader.write("LLM 🤖 (txt) : ", response.get_text_content())
        reader.write("LLM 🤖 (raw) : ", "\n".join([str(msg.to_plain()) for msg in response.messages]))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
