/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import "dotenv/config.js";
import { Backend } from "beeai-framework/backend/core";

const backend = await Backend.fromProvider("ollama");
console.info(backend.chat.modelId);
console.info(backend.embedding.modelId);
