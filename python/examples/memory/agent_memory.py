import asyncio
import sys
import traceback

from beeai_framework.agents.react.agent import ReActAgent
from beeai_framework.agents.types import AgentExecutionConfig
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import AssistantMessage, UserMessage
from beeai_framework.errors import FrameworkError
from beeai_framework.memory.unconstrained_memory import UnconstrainedMemory

# Initialize the memory and LLM
memory = UnconstrainedMemory()


def create_agent() -> ReActAgent:
    llm = ChatModel.from_name("ollama:granite3.1-dense:8b")

    # Initialize the agent
    agent = ReActAgent(llm=llm, memory=memory, tools=[])

    return agent


async def main() -> None:
    # Create user message
    user_input = "Hello world!"
    user_message = UserMessage(user_input)

    # Await adding user message to memory
    await memory.add(user_message)
    print("Added user message to memory")

    # Create agent
    agent = create_agent()

    response = await agent.run(
        prompt=user_input,
        execution=AgentExecutionConfig(max_retries_per_step=3, total_max_retries=10, max_iterations=20),
    )
    print(f"Received response: {response}")

    # Create and store assistant's response
    assistant_message = AssistantMessage(response.result.text)

    # Await adding assistant message to memory
    await memory.add(assistant_message)
    print("Added assistant message to memory")

    # Print results
    print(f"\nMessages in memory: {len(agent.memory.messages)}")

    if len(agent.memory.messages) >= 1:
        user_msg = agent.memory.messages[0]
        print(f"User: {user_msg.text}")

    if len(agent.memory.messages) >= 2:
        agent_msg = agent.memory.messages[1]
        print(f"Agent: {agent_msg.text}")
    else:
        print("No agent message found in memory")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
