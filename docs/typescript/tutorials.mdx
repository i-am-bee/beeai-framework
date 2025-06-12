# BeeAI Framework Tutorials

This repository contains tutorials demonstrating the usage of the BeeAI Framework, a toolkit for building AI agents and applications.

## Table of Contents

1. [How to Slack with Bee](#how-to-slack-with-bee)

## How to Slack with Bee

This tutorial will guide you through integrating the BeeAI Framework with the Slack API. By the end, the agent will be able to post messages to a Slack channel.

### Prerequisites

- [node.js](https://nodejs.org/) v18 or higher
- [ollama](https://ollama.com/) with `llama3.1` model (ensure the model is pulled in advance)

### Setup

1. Clone the starter repository for an easier setup:

```bash
git clone https://github.com/i-am-bee/beeai-framework-starter.git
```

2. Navigate to the repository:

```bash
cd beeai-framework-starter
```

3. Install starter dependencies, including the MCP peer dependency:

```bash
npm i @modelcontextprotocol/sdk
```

4. Create a new agent module:

```bash
touch src/agent_slack.ts
```

That's it for the setup! We’ll add the necessary code to the module you just created.

### Slack API

To connect to the Slack API, you will need a set of credentials and some additional setup.

1. Go to https://api.slack.com/apps and create a new app named `Bee` (use the "from scratch" option).
2. Once on the app page, select `OAuth & Permissions` from the menu.
3. Go to `Bot Token Scopes` and add `chat:write` scope, that will suffice for our purposes.
4. Click `Install to Workspace` and grab the `Bot User OAuth Token`
5. Grab the `Team ID` by navigating to `https://<your-workspace>.slack.com`, after redirect, your URL will change to `https://app.slack.com/client/TXXXXXXX/CXXXXXXX`, pick the segment starting with `TXXXXXXX`.
6. Finally, in Slack, create a public channel `bee-playground` and add `@Bee` app to it.

### Code

The framework doesn't have any specialized tools for using Slack API. However, it supports tools exposed via Model Context Protocol and performs automatic tool discovery. We will use that to give our agent the capability to post Slack messages.

Now, copy and paste the following code into `agent_slack.ts` module. Then, follow along with the comments for an explanation. Do not forget to substitute your Slack credentials while reading through.

```js
import { MCPTool } from "beeai-framework/tools/mcp";
import { ReActAgent } from "beeai-framework/agents/react/agent";
import { UnconstrainedMemory } from "beeai-framework/memory/unconstrainedMemory";
import { OllamaChatModel } from "beeai-framework/adapters/ollama/backend/chat";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { OpenMeteoTool } from "beeai-framework/tools/weather/openMeteo";

// Start by creating MCP client
const client = new Client(
  {
    name: "bee-client",
    version: "1.0.0",
  },
  {
    capabilities: {}, // Our client doesn't expose any capabilities to the server
  },
);

// Connect the client to the StdioClientTransport
// StdioClientTransport runs the specified command and connects to its I/O
// In our case, the command is an MCP Slack server ran via npx
await client.connect(
  new StdioClientTransport({
    command: "npx",
    args: ["-y", "@modelcontextprotocol/server-slack"],
    env: {
      SLACK_BOT_TOKEN: "<your-slack-bot-token>",
      SLACK_TEAM_ID: "<your-slack-team-id>",
      PATH: process.env["PATH"] ?? "", // if omitted, npx command won't get found
    },
  }),
);
try {
  // Discover Slack tools via MCP client
  const slackTools = await MCPTool.fromClient(client);

  // Filter for specific Slack tool
  const filteredSlackTools = slackTools.filter((tool) => tool.name === "slack_post_message");

  // Create ReActAgent agent
  const agent = new ReActAgent({
    // We're using LLM ran locally via Ollama
    llm: new OllamaChatModel("llama3.1"),
    // Besides the Slack tools, we also provide DDG tool for web search
    tools: [new OpenMeteoTool(), ...filteredSlackTools],
    memory: new UnconstrainedMemory(),
    templates: {
      system: (template) =>
        template.fork((config) => {
          config.defaults.instructions = `You are a helpful assistant. When prompted to post to Slack, send messages to the "bee-playground" channel.`;
        }),
    },
  });

  // Execute the agent with prompt
  await agent
    .run({
      // Instruct the agent to send a weather-related question to the Slack
      prompt: "Post to Slack the current temperature in LA.",
    })
    .observe((emitter) => {
      // Track agent progress for debugging
      emitter.on("update", async ({ update }) => {
        console.log(`Agent (${update.key}) 🤖 : `, update.value);
      });
    });
} finally {
  await client.close();
}
```

Run the agent with:

```bash
npm run start src/agent_slack.ts
```

### Conclusion

That's it! You can watch the agent executing in the terminal. Eventually, it should use the `slack_post_message` tool, and the answer should appear in the Slack channel.

As you might have noticed, we made some restrictions to make the agent work with smaller models so that it can be executed locally. With larger LLMs, we could further simplify the code, use more tools, and create simpler prompts.

This tutorial can be easily generalized to any MCP server with tools capability. Just plug it into Bee and execute.
