# Agents

AI agents built on large language models control the path to solving a complex problem. They can typically act on feedback to refine their plan of action, a capability that can improve performance and help them accomplish more sophisticated tasks.

We recommend reading the [following article](https://research.ibm.com/blog/what-are-ai-agents-llm) to learn more.

## Implementation in BeeAI Framework

An agent can be thought of as a program powered by LLM. The LLM generates structured output that is then processed by your program.

Your program then decides what to do next based on the retrieved content. It may leverage a tool, reflect, or produce a final answer.
Before the agent determines the final answer, it performs a series of `steps`. A step might be calling an LLM, parsing the LLM output, or calling a tool.

Steps are grouped in a `iteration`, and every update (either complete or partial) is emitted to the user.

### Bee Agent

Our Bee Agent is based on the `ReAct` ([Reason and Act](https://arxiv.org/abs/2210.03629)) approach.

Hence, the agent in each iteration produces one of the following outputs.

For the sake of simplicity, imagine that the input prompt is "What is the current weather in Las Vegas?"

First iteration:

```
thought: I need to retrieve the current weather in Las Vegas. I can use the OpenMeteo function to get the current weather forecast for a location.
tool_name: OpenMeteo
tool_input: {"location": {"name": "Las Vegas"}, "start_date": "2024-10-17", "end_date": "2024-10-17", "temperature_unit": "celsius"}
```

> [!NOTE]
>
> Agent emitted 3 complete updates in the following order (`thought`, `tool_name`, `tool_input`) and tons of partial updates in the same order.
> Partial update means that new tokens are being added to the iteration. Updates are always in strict order: You first get many partial updates for thought, followed by a final update for thought (that means no final updates are coming for a given key).

Second iteration:

```
thought: I have the current weather in Las Vegas in Celsius.
final_answer: The current weather in Las Vegas is 20.5°C with an apparent temperature of 18.3°C.
```

For more complex tasks, the agent may do way more iterations.

In the following example, we will transform the knowledge gained into code.

```ts
import { ReActAgent } from "beeai-framework/agents/react/agent";
import { OllamaChatModel } from "beeai-framework/adapters/ollama/backend/chat";
import { UnconstrainedMemory } from "beeai-framework/memory/unconstrainedMemory";
import { OpenMeteoTool } from "beeai-framework/tools/weather/openMeteo";

const agent = new ReActAgent({
  llm: new OllamaChatModel("llama3.1"), // for more explore 'beeai-framework/adapters'
  memory: new UnconstrainedMemory(), // for more explore 'beeai-framework/memory'
  tools: [new OpenMeteoTool()], // for more explore 'beeai-framework/tools'
});

const response = await agent
  .run({ prompt: "What is the current weather in Las Vegas?" })
  .observe((emitter) => {
    emitter.on("update", async ({ data, update, meta }) => {
      // to log only valid runs (no errors), check if meta.success === true
      console.log(`Agent Update (${update.key}) 🤖 : ${update.value}`);
      console.log("-> Iteration state", data);
    });

    emitter.on("partialUpdate", async ({ data, update, meta }) => {
      // to log only valid runs (no errors), check if meta.success === true
      console.log(`Agent Partial Update (${update.key}) 🤖 : ${update.value}`);
      console.log("-> Iteration state", data);
    });

    // you can observe other events such as "success" / "retry" / "error" / "toolStart" / "toolEnd", ...

    // To see all events, uncomment the following code block
    // emitter.match("*.*", async (data: unknown, event) => {
    //   const serializedData = JSON.stringify(data).substring(0, 128); // show only part of the event data
    //   console.trace(`Received event "${event.path}"`, serializedData);
    // });
  });

console.log(`Agent: ${response.result.text}`);
```

### Behaviour

You can alter the agent's behavior in the following ways.

#### Setting execution policy

```ts
await agent.run(
  { prompt: "What is the current weather in Las Vegas?" },

  {
    signal: AbortSignal.timeout(60 * 1000), // 1 minute timeout
    execution: {
      // How many times an agent may repeat the given step before it halts (tool call, llm call, ...)
      maxRetriesPerStep: 3,

      // How many retries can the agent occur before halting
      totalMaxRetries: 10,

      // Maximum number of iterations in which the agent must figure out the final answer
      maxIterations: 20,
    },
  },
);
```

> [!NOTE]
>
> The default is zero retries and no timeout.

##### Overriding prompt templates

The agent uses the following prompt templates.

1. **System Prompt**

2. **User Prompt** (to reformat the user's prompt)

3. **User Empty Prompt**

4. **Tool Error**

5. **Tool Input Error** (validation error)

6. **Tool No Result Error**

7. **Tool Not Found Error**

8. **Invalid Schema Error** (output from LLM cannot be processed)

Please refer to the [following example](/typescript/examples/agents/bee_advanced.ts) to see how to modify them.

## Creating your own agent

To create your own agent, you must implement the agent's base class (`BaseAgent`).

The example can be found [here](/typescript/examples/agents/custom_agent.ts).
