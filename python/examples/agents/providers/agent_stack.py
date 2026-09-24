import asyncio
import sys
import traceback

from beeai_framework.adapters.agentstack.agents import AgentStackAgent
from beeai_framework.errors import FrameworkError
from beeai_framework.memory import UnconstrainedMemory
from examples.helpers.io import ConsoleReader


async def main() -> None:
    reader = ConsoleReader()

    agents = await AgentStackAgent.from_agent_stack(url="http://127.0.0.1:8333", memory=UnconstrainedMemory())
    if len(agents) > 1:
        reader.write("Prompt: ", "Select one of the available agents:\n")
        while True:
            for index, agent in enumerate(agents):
                reader.write("AgentStack: ", f"{index}) {agent.name} - {agent.meta.description}")

            agents_index = reader.ask_single_question("Write agent's number: ")
            try:
                agent = agents[int(agents_index)]
                if agent:
                    break

            except (ValueError, IndexError):
                reader.write(
                    "AgentStack (error) : ",
                    f"Invalid selection: `{agents_index}`. Please enter a valid agent number.\n",
                )
    elif len(agents) == 1:
        agent = agents[0]
    else:
        reader.write("AgentStack (error) : ", "No agent registered within the agent stack.\n")
        exit(0)

    reader.write("AgentStack: ", f"Selected {agent.name}:\n")
    for prompt in reader:
        # Run the agent and observe events
        response = await agent.run(prompt).on(
            "update",
            lambda data, event: (reader.write(f"{agent.name} 🤖 (debug) : ", data)),
        )

        reader.write(f"{agent.name} Agent 🤖 : ", response.last_message.text)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
