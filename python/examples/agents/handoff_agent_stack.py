import asyncio

from beeai_framework.adapters.agentstack.agents import AgentStackAgent
from beeai_framework.adapters.agentstack.agents.types import AgentStackAgentStatus
from beeai_framework.adapters.agentstack.backend.chat import AgentStackChatModel
from beeai_framework.adapters.agentstack.serve.server import AgentStackServer
from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.handoff import HandoffTool


async def main() -> None:
    agents = await AgentStackAgent.from_agent_stack(states={AgentStackAgentStatus.ONLINE})
    main_agent = RequirementAgent(
        llm=AgentStackChatModel(),
        name="ManagerAgent",
        instructions="Always delegate task to a sub-agent",
        tools=[HandoffTool(agent) for agent in agents],
        memory=UnconstrainedMemory(),
    )

    server = AgentStackServer()
    server.register(main_agent)
    await server.aserve()


if __name__ == "__main__":
    asyncio.run(main())
