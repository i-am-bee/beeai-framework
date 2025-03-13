import asyncio
import json
import sys
import traceback

from beeai_framework.agents.experimental.remote.agent import RemoteAgent
from beeai_framework.errors import FrameworkError
from examples.helpers.io import ConsoleReader


async def main() -> None:
    reader = ConsoleReader()

    agent = RemoteAgent(url="http://127.0.0.1:8333/mcp/sse", agent="chat")
    for prompt in reader:
        # Run the agent and observe events
        response = await agent.run(
            prompt=f'{{"messages":[{{"role":"user","content":"{prompt}"}}],"config":{{"tools":["weather","search"]}}}}'
        ).on(
            "update",
            lambda data, event: (
                reader.write("Agent 🤖 : ", data["update"]["value"]["logs"][0]["message"])
                if "logs" in data["update"]["value"]
                else None
            ),
        )

        reader.write("Agent 🤖 : ", json.loads(response.result.text)["messages"][0]["content"])


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
