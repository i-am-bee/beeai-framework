import "dotenv/config.js";
import { Backend } from "beeai-framework/backend/core";

const backend = await Backend.fromProvider("ollama");
console.info(backend.chat.modelId);
console.info(backend.embedding.modelId);
