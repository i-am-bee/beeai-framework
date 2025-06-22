## 🤖 RequirementAgent

> [!NOTE]
>
> This is an experimental feature and will evolve based on community feedback.
>
> **Location within the framework:** [`beeai_framework/agents/experimental`](/python/beeai_framework/agents/experimental).


The `RequirementAgent` combines the power of LLMs, tools, and requirements, all wrapped in a declarative interface.

**▶️ Example**

```python
agent = RequirementAgent(
  llm=ChatModel.from_name("ollama:granite3.3:8b"),
  tools=[ThinkTool(), OpenMeteoTool(), DuckDuckGoSearchTool()],
  instructions="Plan activities for a given destination based on current weather and events.",
  requirements=[
      ConditionalRequirement(ThinkTool, force_at_step=1),
      ConditionalRequirement(DuckDuckGoSearchTool, only_after=[OpenMeteoTool], min_invocations=1),
  ],
)
```

```python
response = await agent.run("What to do in Boston?").middleware(GlobalTrajectoryMiddleware())
print(response.answer.text)
```

➡️ Check out more [examples](/python/examples/experimental/requirement) (multi-agent, custom requirements, ...).

## 💬 Motivation

Every language model is different and expects a unique style of communication. This makes it complicated to create a single system prompt (agent) that works well for all models. Nudging the model within existing implementations can be tricky, as system prompts are often highly opinionated and typically optimized for a single model or a small subset. Additionally, an agent’s system prompt may force the LLM to produce text in a predefined structure, which models may not be able to follow. The model might get confused by the large context window or simply be incapable of adhering to the structure.

As a result, you never know whether the response is hallucinated or backed by evidence from a tool. For smaller models, you need to be more explicit, while for more capable models, you can be more flexible. This is why people have started switching to workflows (graphs), which are typically more restrictive but come with additional development costs.

We have addressed these issues in our new agent-building abstraction.

## 🔎 Core principles

- **Everything is a tool**
  - From data retrieval and web search to reasoning or even providing a final answer.
  - This allows us to generate outputs that adhere to a user-defined format (leveraging structured decoding).
  - The model always produces a valid response—no more parsing errors.

- **Tool execution is influenced by a set of requirements**
  - Use tool `A` only after tool `B` was called.
  - Tool `D` must be run exactly twice, but never in a row.
  - Tool `E` can only be run after both `A` and `B` have been executed.
  - Tool `F` must be called immediately after tool `D`.
  - You can't provide a final answer before calling tool `C` at least once.

## 📋 What are Requirements?

**Requirements** are functions that take the current agent state and produce a list of **rules**.

**Rules** are objects that define restrictions to the agent based on its current state:

- The tool the rule applies to (`target` attribute)
- Whether the tool can be used (`allowed` attribute)
- Whether the tool’s definition is visible to the agent (`hidden` attribute)
- Whether the rule blocks termination (`prevent_stop` attribute)
- Whether the tool must be invoked (`forced` attribute)

> [!IMPORTANT]
>
> If two different rules force a tool execution, the one with a higher priority is applied.

> [!IMPORTANT]
>
> Forbidding a tool’s usage takes precedence over allowing it.

> [!NOTE]
>
> Requirements are invoked on every iteration's start (before calling an LLM).

> [!TIP]
>
> Start with a single requirement and add more as needed.

### Conditional

A requirement that enables the use of a given tool (or set of tools) after certain conditions are met.
See the following examples.

**Define order of execution**

In the following example, we force the agent to use `ThinkTool` (for reasoning) followed by `DuckDuckGoSearchTool` to retrieve data. This trajectory ensures that even a small model can arrive at the correct answer. Without such enforcement, the agent might skip tool calls entirely.

```python
RequirementAgent(
  llm=ChatModel.from_name("ollama:granite3.3:8b"),
  tools=[ThinkTool(), DuckDuckGoSearchTool()],
  requirements=[
      ConditionalRequirement(ThinkTool, force_at_step=1), # Force ThinkTool at the first step
      ConditionalRequirement(DuckDuckGoSearchTool, force_at_step=2), # Force DuckDuckGo at the second step
  ],
)
```

**Creating a ReAct agent**

A ReAct Agent (Reason and Act) follows this trajectory:

```text
Think -> Use a tool -> Think -> Use a tool -> Think -> ... -> Final Answer
```

This can be easily achieved by forcing the execution of the `Think` tool after every other tool.

```python
RequirementAgent(
  llm=ChatModel.from_name("ollama:granite3.3:8b"),
  tools=[ThinkTool(), WikipediaTool(), OpenMeteoTool()],
  requirements=[ConditionalRequirement(ThinkTool, force_at_step=1, force_after=[OpenMeteoTool, WikipediaTool])],
)
```

> [!TIP]
>
> To generalize further, use `ConditionalRequirement(ThinkTool, force_at_step=1, force_after=Tool, consecutive_allowed=False)`, where the option `consecutive_allowed=False` simply denotes that `ThinkTool` cannot be used multiple times in a row.

**Creating a ReAct agent with custom conditions**

You may want an agent that works like ReAct but skips the 'reasoning' (thinking) step under certain conditions. In the example below, the `priority` option is used to tell the agent to send an email by calling the `send_email` tool after `create_order` is invoked, and call `ThinkTool` after every other action.

```python
RequirementAgent(
  llm=ChatModel.from_name("ollama:granite3.3:8b"),
  tools=[ThinkTool(), retrieve_basket(), create_order(), send_email()],
  requirements=[
    ConditionalRequirement(ThinkTool, force_at_step=1, force_after=Tool, priority=10),
    ConditionalRequirement(send_email, only_after=create_order, force_after=create_order, priority=20, max_invocations=1),
  ],
)
```

**Prevent agent termination if a tool hasn't been called yet**

The following requirement prevents the agent from providing a final answer before it calls the `my_tool`.

```python
ConditionalRequirement(my_tool, min_invocations=1)
```

**List of possible arguments**

```python
ConditionalRequirement(
  target_tool, # Tool class, instance, or name (can also be specified as `target=...`)
  name="", # (optional) Name, useful for logging
  only_before=[...], # (optional) Disable target_tool after any of these tools are called
  only_after=[...], # (optional) Disable target_tool before all these tools are called
  force_after=[...], # (optional) Force target_tool execution immediately after any of these tools are called
  min_invocations=0, # (optional) Minimum times the tool must be called before agent can stop
  max_invocations=10, # (optional) Maximum times the tool can be called before being disabled
  force_at_step=1, # (optional) Step number at which the tool must be invoked
  only_success_invocations=True, # (optional) Whether 'force_at_step' counts only successful invocations
  priority=10, # (optional) Higher number means higher priority for requirement enforcement
  consecutive_allowed=True, # (optional) Whether the tool can be invoked twice in a row
  enabled=True, # (optional) Whether to skip this requirement’s execution
  custom_checks=[
     # (optional) Custom callbacks; all must pass for the tool to be used
    lambda state: any('weather' in msg.text for msg in state.memory.message if isinstance(msg, UserMessage)),
    lambda state: state.iteration > 0,
  ],
)
```

> [!TIP]
>
> It is recommended to pass a class instance (e.g., `weather_tool = ...`) or a class (`OpenMeteoTool`) rather than a tool's name, as some tools may have dynamically generated names.

> [!NOTE]
>
> If the reasoner detects two contradictory rules or a rule without an existing target, it throws an error.

### AskPermission

Some tools may be expensive to run or have destructive effects. For these, you may want to get approval from an external system or directly from the user.

```python
RequirementAgent(
  llm=ChatModel.from_name("ollama:granite3.3:8b"),
  tools=[get_data, remove_data, update_data],
  requirements=[AskPermissionRequirement([remove_data, get_data])]
)
```

**Using a custom source**

```python
async def handler(tool: Tool, input: dict[str, Any]) -> bool:
  # your implementation
  return True

AskPermissionRequirement(..., handler=handler)
```

**All possible parameters**

```python
AskPermissionRequirement(
    include=[...], # (optional) List of targets (tool name, instance, or class) requiring explicit approval
    exclude=[...], # (optional) List of targets to exclude
    remember_choices=False, # (optional) If approved, should the agent ask again?
    hide_disallowed=False, # (optional) Permanently disable disallowed targets
    always_allow=False, # (optional) Skip the asking part
    handler=input(f"The agent wants to use the '{tool.name}' tool.\nInput: {tool_input}\nDo you allow it? (yes/no): ").strip().startswith("yes") # (optional) Custom handler, can be async
)
```

> [!NOTE]
>
> If no targets are specified, permission is required for all tools.

#### Writing Your Own Requirement

You can create a custom requirement by implementing the base `Requirement` class.

The following example demonstrates how to write a requirement that prevents the agent from answering if the question contains a certain phrase.

<!-- embedme examples/agents/experimental/requirement/custom_requirement.py -->

```python
import asyncio

from beeai_framework.agents.experimental import RequirementAgent
from beeai_framework.agents.experimental.requirements import Requirement, Rule
from beeai_framework.agents.experimental.requirements.requirement import run_with_context
from beeai_framework.agents.experimental.types import RequirementAgentRunState
from beeai_framework.backend import AssistantMessage, ChatModel
from beeai_framework.context import RunContext
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool


class PrematureStopRequirement(Requirement[RequirementAgentRunState]):
    """Prevents the agent from answering if a certain phrase occurs in the conversation"""

    name = "premature_stop"

    def __init__(self, phrase: str) -> None:
        super().__init__()
        self._phrase = phrase
        self._priority = 100  # (optional), default is 10

    @run_with_context
    async def run(self, input: RequirementAgentRunState, context: RunContext) -> list[Rule]:
        last_message = input.memory.messages[-1]
        if self._phrase in last_message.text:
            await input.memory.add(
                AssistantMessage(
                    "The final answer is that the system policy does not allow me to answer this type of questions.",
                    {"tempMessage": True},  # the message gets removed in the next iteration
                )
            )
            return [Rule(target="final_answer", forced=True)]
        else:
            return []


async def main() -> None:
    agent = RequirementAgent(
        llm=ChatModel.from_name("ollama:granite3.3:8b"),
        tools=[DuckDuckGoSearchTool()],
        requirements=[PrematureStopRequirement("value of x")],
    )
    prompt = "y = 2x + 4, what is the value of x?"
    print("👤 User: ", prompt)
    response = await agent.run(prompt).middleware(GlobalTrajectoryMiddleware())
    print("🤖 Agent: ", response.answer.text)


if __name__ == "__main__":
    asyncio.run(main())

```

_Source: [`python/examples/agents/experimental/requirement/custom_requirement.py`](/python/examples/agents/experimental/requirement/custom_requirement.py)_
