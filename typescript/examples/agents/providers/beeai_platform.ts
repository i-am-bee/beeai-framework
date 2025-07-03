/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import "dotenv/config.js";
import { BeeAIPlatformAgent } from "beeai-framework/adapters/beeai_platform/agents/agent";
import { createConsoleReader } from "examples/helpers/io.js";
import { FrameworkError } from "beeai-framework/errors";
import { TokenMemory } from "beeai-framework/memory/tokenMemory";

const agentName = "chat";

const instance = new BeeAIPlatformAgent({
  url: "http://127.0.0.1:8333/api/v1/acp",
  agentName,
  memory: new TokenMemory(),
});

const reader = createConsoleReader();

try {
  for await (const { prompt } of reader) {
    const result = await instance.run({ input: prompt }).observe((emitter) => {
      emitter.on("update", (data) => {
        reader.write(`Agent (received progress) 🤖 : `, JSON.stringify(data.value, null, 2));
      });
      emitter.on("error", (data) => {
        reader.write(`Agent (error) 🤖 : `, data.message);
      });
    });

    reader.write(`Agent (${agentName}) 🤖 : `, result.result.text);
  }
} catch (error) {
  reader.write("Agent (error)  🤖", FrameworkError.ensure(error).dump());
}
