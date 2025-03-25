/**
 * Copyright 2025 IBM Corp.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

export const BackendProviders = {
  OpenAI: { name: "OpenAI", module: "openai", aliases: ["openai"] as string[] },
  Azure: { name: "Azure", module: "azure", aliases: ["microsoft", "microsoft-azure"] as string[] },
  Watsonx: { name: "Watsonx", module: "watsonx", aliases: ["watsonx", "ibm"] as string[] },
  Ollama: { name: "Ollama", module: "ollama", aliases: [] as string[] },
  GoogleVertex: {
    name: "GoogleVertex",
    module: "google-vertex",
    aliases: ["google", "vertex"] as string[],
  },
  Bedrock: {
    name: "Bedrock",
    module: "amazon-bedrock",
    aliases: ["amazon", "bedrock"] as string[],
  },
  Groq: { name: "Groq", module: "groq", aliases: [] as string[] },
  Dummy: { name: "Dummy", module: "dummy", aliases: [] as string[] },
  Anthropic: { name: "Anthropic", module: "anthropic", aliases: [] as string[] },
} as const;

export type ProviderName = (typeof BackendProviders)[keyof typeof BackendProviders]["module"];
export type ProviderHumanName = (typeof BackendProviders)[keyof typeof BackendProviders]["name"];
export type ProviderDef = (typeof BackendProviders)[keyof typeof BackendProviders];
