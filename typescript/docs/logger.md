# Logger

> [!TIP]
>
> Location within the framework `beeai-framework/logger`.

The Logger is a key component designed to record and track events, errors, and other important actions during an application's execution. It provides valuable insights into the application's behavior, performance, and potential issues, helping developers and system administrators troubleshoot and monitor the system effectively.

In the BeeAI Framework, the [Logger](/typescript/src/logger/logger.ts) class is an abstraction built on top of the popular [pino](https://github.com/pinojs/pino) logger, offering additional flexibility and integration.

## Basic Usage

<!-- embedme examples/logger/base.ts -->

```ts
import { Logger, LoggerLevel } from "beeai-framework/logger/logger";

// Configure logger defaults
Logger.defaults.pretty = true; // Pretty-print logs (default: false, can also be set via ENV: BEE_FRAMEWORK_LOG_PRETTY=true)
Logger.defaults.level = LoggerLevel.TRACE; // Set log level to trace (default: TRACE, can also be set via ENV: BEE_FRAMEWORK_LOG_LEVEL=trace)
Logger.defaults.name = undefined; // Optional name for logger (default: undefined)
Logger.defaults.bindings = {}; // Optional bindings for structured logging (default: empty)

// Create a child logger for your app
const logger = Logger.root.child({ name: "app" });

// Log at different levels
logger.trace("Trace!");
logger.debug("Debug!");
logger.info("Info!");
logger.warn("Warning!");
logger.error("Error!");
logger.fatal("Fatal!");
```

_Source: [examples/logger/base.ts](/typescript/examples/logger/base.ts)_

## Usage with Agents

The [Logger](/typescript/src/logger/logger.ts) seamlessly integrates with agents in the framework. Below is an example that demonstrates how logging can be used in conjunction with agents and event emitters.

<!-- embedme examples/logger/agent.ts -->

```ts
import { ReActAgent } from "beeai-framework/agents/react/agent";
import { UnconstrainedMemory } from "beeai-framework/memory/unconstrainedMemory";
import { Logger } from "beeai-framework/logger/logger";
import { Emitter } from "beeai-framework/emitter/emitter";
import { OllamaChatModel } from "beeai-framework/adapters/ollama/backend/chat";

// Set up logging
Logger.defaults.pretty = true;

const logger = Logger.root.child({
  level: "trace",
  name: "app",
});

// Log events emitted during agent execution
Emitter.root.match("*.*", (data, event) => {
  const logLevel = event.path.includes(".run.") ? "trace" : "info";
  logger[logLevel](`Event '${event.path}' triggered by '${event.creator.constructor.name}'`);
});

// Create and run an agent
const agent = new ReActAgent({
  llm: new OllamaChatModel("llama3.1"),
  memory: new UnconstrainedMemory(),
  tools: [],
});

const response = await agent.run({ prompt: "Hello!" });
logger.info(response.result.text);
```

_Source: [examples/logger/agent.ts](/typescript/examples/logger/agent.ts)_

## Custom pino instance integration

If you need to integrate your own `pino` instance with the BeeAI Framework Logger, you can do so easily. Below is an example that demonstrates how to create a pino logger and use it with the framework’s [Logger](/typescript/src/logger/logger.ts).

<!-- embedme examples/logger/pino.ts -->

```ts
import { Logger } from "beeai-framework/logger/logger";
import { pino } from "pino";

// Create a custom pino logger
const customLogger = pino({
  name: "app",
});

// Use the custom pino instance within the framework
const frameworkLogger = new Logger(
  {
    level: "info", // Set the log level
    name: "framework", // Set the logger name
  },
  customLogger, // Pass the custom pino instance
);
```

_Source: [examples/logger/pino.ts](/typescript/examples/logger/pino.ts)_
