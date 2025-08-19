import asyncio

from beeai_framework.agents.tool_calling import ToolCallingAgent
from beeai_framework.agents.tool_calling.prompts import ToolCallingAgentSystemPrompt, ToolCallingAgentTaskPrompt
from beeai_framework.backend import ChatModel
from beeai_framework.context import RunContext


async def main() -> None:
    llm = ChatModel.from_name("ollama:granite3.3:8b")
    llm.emitter.on("start", lambda data, event: data.input.tools.pop(0))
    agent = ToolCallingAgent(
        llm=llm,
        final_answer_as_tool=False,
        templates={
            "system": ToolCallingAgentSystemPrompt.fork(
                lambda model: model.model_copy(update={"template": "You are a helpful assistant."})
            ),
            "task": ToolCallingAgentTaskPrompt.fork(lambda model: model.model_copy(update={"template": "{{prompt}}"})),
        },
    )

    emitter.on(lambda event: event.name == "start" and isinstance(event.creator, Tool), lambda data, event: print(event.creator.name))
    emitter.on(lambda event: event.name == "finish" and isinstance(event.creator, Tool), lambda data, event: print(event.creator.name))


    ll = agent.run("...").context({"xxx": 42})

    ctx = RunContext.get()
    print(ctx)


if __name__ == "__main__":
    asyncio.run(main())
