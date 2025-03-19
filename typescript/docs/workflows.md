# Workflows (experimental)

> [!TIP]
>
> Location within the framework `beeai-framework/experimental/workflows`.

Workflows provide a flexible and extensible component for managing and executing structured sequences of tasks.

- Dynamic Execution: Steps can direct the flow based on state or results.
- Validation: Define schemas for data consistency and type safety.
- Modularity: Steps can be standalone or invoke nested workflows.
- Observability: Emit events during execution to track progress or handle errors.

## Usage

#### Basic

<!-- embedme examples/workflows/simple.ts -->

```ts
import { Workflow } from "beeai-framework/workflows/workflow";
import { z } from "zod";

const schema = z.object({
  hops: z.number().default(0),
});

const workflow = new Workflow({ schema })
  .addStep("a", async (state) => {
    state.hops += 1;
  })
  .addStep("b", () => (Math.random() > 0.5 ? Workflow.PREV : Workflow.END));

const response = await workflow.run({ hops: 0 }).observe((emitter) => {
  emitter.on("start", (data) => console.log(`-> start ${data.step}`));
  emitter.on("error", (data) => console.log(`-> error ${data.step}`));
  emitter.on("success", (data) => console.log(`-> finish ${data.step}`));
});

console.log(`Hops: ${response.result.hops}`);
console.log(`-> steps`, response.steps.map((step) => step.name).join(","));
```

_Source: [examples/workflows/simple.ts](/typescript/examples/workflows/simple.ts)_

#### Nesting

<!-- embedme examples/workflows/nesting.ts -->

```ts
import { Workflow } from "beeai-framework/workflows/workflow";
import { z } from "zod";

const schema = z.object({
  threshold: z.number().min(0).max(1),
  counter: z.number().default(0),
});

const addFlow = new Workflow({ schema }).addStep("run", async (state) => {
  state.counter += 1;
  return Math.random() > 0.5 ? Workflow.SELF : Workflow.END;
});

const subtractFlow = new Workflow({
  schema,
}).addStep("run", async (state) => {
  state.counter -= 1;
  return Math.random() > 0.5 ? Workflow.SELF : Workflow.END;
});

const workflow = new Workflow({
  schema,
})
  .addStep("start", (state) =>
    Math.random() > state.threshold ? "delegateAdd" : "delegateSubtract",
  )
  .addStep("delegateAdd", addFlow.asStep({ next: Workflow.END }))
  .addStep("delegateSubtract", subtractFlow.asStep({ next: Workflow.END }));

const response = await workflow.run({ threshold: 0.5 }).observe((emitter) => {
  emitter.on("start", (data, event) =>
    console.log(`-> step ${data.step}`, event.trace?.parentRunId ? "(nested flow)" : ""),
  );
});
console.info(`Counter:`, response.result);
```

_Source: [examples/workflows/nesting.ts](/typescript/examples/workflows/nesting.ts)_

### Agent Delegation

<!-- embedme examples/workflows/agent.ts -->

```ts
import "dotenv/config";
import { ReActAgent } from "beeai-framework/agents/react/agent";
import { z } from "zod";
import { Message, UserMessage } from "beeai-framework/backend/message";
import { WikipediaTool } from "beeai-framework/tools/search/wikipedia";
import { OpenMeteoTool } from "beeai-framework/tools/weather/openMeteo";
import { ReadOnlyMemory } from "beeai-framework/memory/base";
import { UnconstrainedMemory } from "beeai-framework/memory/unconstrainedMemory";
import { Workflow } from "beeai-framework/workflows/workflow";
import { createConsoleReader } from "examples/helpers/io.js";
import { GroqChatModel } from "beeai-framework/adapters/groq/backend/chat";

const schema = z.object({
  answer: z.instanceof(Message).optional(),
  memory: z.instanceof(ReadOnlyMemory),
});

const workflow = new Workflow({ schema: schema })
  .addStep("simpleAgent", async (state) => {
    const simpleAgent = new ReActAgent({
      llm: new GroqChatModel("llama-3.3-70b-versatile"),
      tools: [],
      memory: state.memory,
    });
    const answer = await simpleAgent.run({ prompt: null });
    reader.write("🤖 Simple Agent", answer.result.text);

    state.answer = answer.result;
    return "critique";
  })
  .addStrictStep("critique", schema.required(), async (state) => {
    const llm = new GroqChatModel("llama-3.3-70b-versatile");
    const { object: critiqueResponse } = await llm.createStructure({
      schema: z.object({ score: z.number().int().min(0).max(100) }),
      messages: [
        Message.of({
          role: "system",
          text: `You are an evaluation assistant who scores the credibility of the last assistant's response. Chitchatting always has a score of 100. If the assistant was unable to answer the user's query, then the score will be 0.`,
        }),
        ...state.memory.messages,
        state.answer,
      ],
    });
    reader.write("🧠 Score", critiqueResponse.score.toString());

    return critiqueResponse.score < 75 ? "complexAgent" : Workflow.END;
  })
  .addStep("complexAgent", async (state) => {
    const complexAgent = new ReActAgent({
      llm: new GroqChatModel("llama-3.3-70b-versatile"),
      tools: [new WikipediaTool(), new OpenMeteoTool()],
      memory: state.memory,
    });
    const { result } = await complexAgent.run({ prompt: null });
    reader.write("🤖 Complex Agent", result.text);
    state.answer = result;
  })
  .setStart("simpleAgent");

const reader = createConsoleReader();
const memory = new UnconstrainedMemory();

for await (const { prompt } of reader) {
  const userMessage = new UserMessage(prompt);
  await memory.add(userMessage);

  const response = await workflow.run({
    memory: memory.asReadOnly(),
  });
  await memory.add(response.state.answer!);

  reader.write("🤖 Final Answer", response.state.answer!.text);
}
```

_Source: [examples/workflows/agent.ts](/typescript/examples/workflows/agent.ts)_

### Multi-agent Content Creator

<!-- embedme examples/workflows/contentCreator.ts -->

```ts
import "dotenv/config";
import { z } from "zod";
import { Workflow } from "beeai-framework/workflows/workflow";
import { ReActAgent } from "beeai-framework/agents/react/agent";
import { UnconstrainedMemory } from "beeai-framework/memory/unconstrainedMemory";
import { createConsoleReader } from "examples/helpers/io.js";
import { Message } from "beeai-framework/backend/message";
import { isEmpty } from "remeda";
import { LLMTool } from "beeai-framework/tools/llm";
import { GoogleSearchTool } from "beeai-framework/tools/search/googleSearch";
import { GroqChatModel } from "beeai-framework/adapters/groq/backend/chat";

const schema = z.object({
  input: z.string(),
  output: z.string().optional(),

  topic: z.string().optional(),
  notes: z.array(z.string()).default([]),
  plan: z.string().optional(),
  draft: z.string().optional(),
});

const workflow = new Workflow({
  schema,
  outputSchema: schema.required({ output: true }),
})
  .addStep("preprocess", async (state) => {
    const llm = new GroqChatModel("llama-3.3-70b-versatile");

    const { object: parsed } = await llm.createStructure({
      schema: schema.pick({ topic: true, notes: true }).or(
        z.object({
          error: z
            .string()
            .describe("Use when the input query does not make sense or you need clarification."),
        }),
      ),
      messages: [
        Message.of({
          role: `user`,
          text: [
            "Your task is to rewrite the user query so that it guides the content planner and editor to craft a blog post that perfectly aligns with the user's needs. Notes should be used only if the user complains about something.",
            "If the user query does ",
            "",
            ...[state.topic && ["# Previous Topic", state.topic, ""]],
            ...[!isEmpty(state.notes) && ["# Previous Notes", ...state.notes, ""]],
            "# User Query",
            state.input || "empty",
          ]
            .filter(Boolean)
            .join("\n"),
        }),
      ],
    });

    if ("error" in parsed) {
      state.output = parsed.error;
      return Workflow.END;
    }

    state.notes = parsed.notes ?? [];
    state.topic = parsed.topic;
  })
  .addStrictStep("planner", schema.required({ topic: true }), async (state) => {
    const llm = new GroqChatModel("llama-3.3-70b-versatile");
    const agent = new ReActAgent({
      llm,
      memory: new UnconstrainedMemory(),
      tools: [new GoogleSearchTool(), new LLMTool({ llm })],
    });

    agent.emitter.on("update", (data) => {
      console.info(data.update);
    });

    const { result } = await agent.run({
      prompt: [
        `You are a Content Planner. Your task is to write a content plan for "${state.topic}" topic in Markdown format.`,
        ``,
        `# Objectives`,
        `1. Prioritize the latest trends, key players, and noteworthy news.`,
        `2. Identify the target audience, considering their interests and pain points.`,
        `3. Develop a detailed content outline including introduction, key points, and a call to action.`,
        `4. Include SEO keywords and relevant sources.`,
        ``,
        ...[!isEmpty(state.notes) && ["# Notes", ...state.notes, ""]],
        `Provide a structured output that covers the mentioned sections.`,
      ].join("\n"),
    });

    state.plan = result.text;
  })
  .addStrictStep("writer", schema.required({ plan: true }), async (state) => {
    const llm = new GroqChatModel("llama-3.3-70b-versatile");
    const output = await llm.create({
      messages: [
        Message.of({
          role: `system`,
          text: [
            `You are a Content Writer. Your task is to write a compelling blog post based on the provided context.`,
            ``,
            `# Context`,
            `${state.plan}`,
            ``,
            `# Objectives`,
            `- An engaging introduction`,
            `- Insightful body paragraphs (2-3 per section)`,
            `- Properly named sections/subtitles`,
            `- A summarizing conclusion`,
            `- Format: Markdown`,
            ``,
            ...[!isEmpty(state.notes) && ["# Notes", ...state.notes, ""]],
            `Ensure the content flows naturally, incorporates SEO keywords, and is well-structured.`,
          ].join("\n"),
        }),
      ],
    });

    state.draft = output.getTextContent();
  })
  .addStrictStep("editor", schema.required({ draft: true }), async (state) => {
    const llm = new GroqChatModel("llama-3.3-70b-versatile");
    const output = await llm.create({
      messages: [
        Message.of({
          role: `system`,
          text: [
            `You are an Editor. Your task is to transform the following draft blog post to a final version.`,
            ``,
            `# Draft`,
            `${state.draft}`,
            ``,
            `# Objectives`,
            `- Fix Grammatical errors`,
            `- Journalistic best practices`,
            ``,
            ...[!isEmpty(state.notes) && ["# Notes", ...state.notes, ""]],
            ``,
            `IMPORTANT: The final version must not contain any editor's comments.`,
          ].join("\n"),
        }),
      ],
    });

    state.output = output.getTextContent();
  });

let lastResult = {} as Workflow.output<typeof workflow>;
const reader = createConsoleReader();
for await (const { prompt } of reader) {
  const { result } = await workflow
    .run({
      input: prompt,
      notes: lastResult?.notes,
      topic: lastResult?.topic,
    })
    .observe((emitter) => {
      emitter.on("start", ({ step, run }) => {
        reader.write(`-> ▶️ ${step}`, JSON.stringify(run.state).substring(0, 200).concat("..."));
      });
    });

  lastResult = result;
  reader.write("🤖 Answer", lastResult.output);
}
```

_Source: [examples/workflows/contentCreator.ts](/typescript/examples/workflows/contentCreator.ts)_

### Multi Agents Workflows

<!-- embedme examples/workflows/multiAgents.ts -->

```ts
import "dotenv/config";
import { createConsoleReader } from "examples/helpers/io.js";
import { OpenMeteoTool } from "beeai-framework/tools/weather/openMeteo";
import { WikipediaTool } from "beeai-framework/tools/search/wikipedia";
import { AgentWorkflow } from "beeai-framework/workflows/agent";
import { OllamaChatModel } from "beeai-framework/adapters/ollama/backend/chat";

const workflow = new AgentWorkflow("Smart assistant");
const llm = new OllamaChatModel("llama3.1");

workflow.addAgent({
  name: "Researcher",
  role: "A diligent researcher",
  instructions: "You look up and provide information about a specific topic.",
  tools: [new WikipediaTool()],
  llm,
});
workflow.addAgent({
  name: "WeatherForecaster",
  role: "A weather reporter",
  instructions: "You provide detailed weather reports.",
  tools: [new OpenMeteoTool()],
  llm,
});
workflow.addAgent({
  name: "DataSynthesizer",
  role: "A meticulous and creative data synthesizer",
  instructions: "You can combine disparate information into a final coherent summary.",
  llm,
});

const reader = createConsoleReader();
reader.write("Assistant 🤖 : ", "What location do you want to learn about?");
for await (const { prompt } of reader) {
  const { result } = await workflow
    .run([
      { prompt: "Provide a short history of the location.", context: prompt },
      {
        prompt: "Provide a comprehensive weather summary for the location today.",
        expectedOutput:
          "Essential weather details such as chance of rain, temperature and wind. Only report information that is available.",
      },
      {
        prompt: "Summarize the historical and weather data for the location.",
        expectedOutput:
          "A paragraph that describes the history of the location, followed by the current weather conditions.",
      },
    ])
    .observe((emitter) => {
      emitter.on("success", (data) => {
        reader.write(
          `Step '${data.step}' has been completed with the following outcome:\n`,
          data.state?.finalAnswer ?? "-",
        );
      });
    });

  reader.write(`Assistant 🤖`, result.finalAnswer);
  reader.write("Assistant 🤖 : ", "What location do you want to learn about?");
}
```

_Source: [examples/workflows/multiAgents.ts](/typescript/examples/workflows/multiAgents.ts)_
