/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

export const BackendProviders = {
  OpenAI: { name: "OpenAI", module: "openai", aliases: ["openai"] as string[] },
  Azure: {
    name: "Azure",
    module: "azure",
    aliases: ["microsoft", "microsoft-azure"] as string[],
  },
  Watsonx: {
    name: "Watsonx",
    module: "watsonx",
    aliases: ["watsonx", "ibm"] as string[],
  },
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
  Xai: { name: "XAI", module: "xai", aliases: ["grok", "xai"] as string[] },
  Dummy: { name: "Dummy", module: "dummy", aliases: [] as string[] },
  Anthropic: {
    name: "Anthropic",
    module: "anthropic",
    aliases: [] as string[],
  },
} as const;

export type ProviderName = (typeof BackendProviders)[keyof typeof BackendProviders]["module"];
export type ProviderHumanName = (typeof BackendProviders)[keyof typeof BackendProviders]["name"];
export type ProviderDef = (typeof BackendProviders)[keyof typeof BackendProviders];
