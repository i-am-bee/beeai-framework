/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { TokenMemory } from "beeai-framework/memory/tokenMemory";
import { AssistantMessage, UserMessage } from "beeai-framework/backend/message";

const memory = new TokenMemory();
await memory.add(new UserMessage("What is your name?"));

const serialized = await memory.serialize();
const deserialized = await TokenMemory.fromSerialized(serialized);

await deserialized.add(new AssistantMessage("Bee"));
