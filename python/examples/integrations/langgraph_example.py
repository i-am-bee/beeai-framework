# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

# To run this example, the optional packages:
#
#    - langchain-core
#    - langchain-community
#    - langchain-ollama
#    - langgraph
#
# need to be installed

import asyncio
import random
import sys
import traceback

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, InstanceOf

from beeai_framework.agents import AgentExecutionConfig
from beeai_framework.agents.react import ReActAgent
from beeai_framework.backend import AnyMessage, AssistantMessage, ChatModel, Role, UserMessage
from beeai_framework.errors import FrameworkError
from beeai_framework.memory import ReadOnlyMemory, UnconstrainedMemory
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool
from beeai_framework.workflows import Workflow, WorkflowRun
from examples.helpers.io import ConsoleReader


def _convert_message(message: AnyMessage) -> BaseMessage:
    if message.role == Role.ASSISTANT:
        return AIMessage(content=message.text)
    elif message.role == Role.SYSTEM:
        return SystemMessage(content=message.text)
    elif message.role == Role.TOOL:
        return ToolMessage(content=message.text)
    else:
        return HumanMessage(content=message.text)


async def main() -> None:
    class Schema(BaseModel):
        memory: InstanceOf[ReadOnlyMemory]
        answer: str = ""

    async def bee_step(state: Schema) -> str:
        agent = ReActAgent(
            llm=ChatModel.from_name("ollama:llama3.1"), tools=[DuckDuckGoSearchTool()], memory=state.memory
        )
        response = await agent.run(execution=AgentExecutionConfig(max_iterations=5))
        state.answer = response.result.text
        return Workflow.END

    def langgraph_step(state: Schema) -> str:
        langgraph_agent = create_react_agent(ChatOllama(model="llama3.1", temperature=0), tools=[DuckDuckGoSearchRun()])
        response = langgraph_agent.invoke(
            {"messages": [_convert_message(msg) for msg in state.memory.messages]},
            RunnableConfig(recursion_limit=5),
        )
        state.answer = "".join([str(msg.content) for msg in response["messages"]])
        return Workflow.END

    workflow = Workflow(Schema)
    workflow.add_step("router", lambda state: random.choice(["bee", "langgraph"]))
    workflow.add_step("bee", bee_step)
    workflow.add_step("langgraph", langgraph_step)

    memory = UnconstrainedMemory()
    reader = ConsoleReader()

    for prompt in reader:
        await memory.add(UserMessage(content=prompt))
        workflow_run: WorkflowRun[Schema] = await workflow.run({"memory": memory.as_read_only()})
        result = workflow_run.result.answer if workflow_run.result else ""
        reader.write("LLM 🤖 : ", result)
        reader.write("-> solved by ", workflow_run.steps[-1].name)
        await memory.add(AssistantMessage(content=result))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
