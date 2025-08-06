# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

from pydantic import BaseModel

ProviderName = Literal[
    "beeai",
    "ollama",
    "openai",
    "watsonx",
    "groq",
    "xai",
    "vertexai",
    "gemini",
    "amazon_bedrock",
    "anthropic",
    "azure_openai",
    "mistralai",
    "langchain",
    "llamaindex",
]
ProviderHumanName = Literal[
    "BeeAI",
    "Ollama",
    "OpenAI",
    "Watsonx",
    "Groq",
    "XAI",
    "VertexAI",
    "Gemini",
    "AmazonBedrock",
    "Anthropic",
    "AzureOpenAI",
    "MistralAI",
    "LangChain",
    "LlamaIndex",
]

ModelTypes = Literal["embedding", "chat"]
ModuleTypes = Literal["vector_store", "document_processor", "document_loader", "text_splitter"]


class ProviderDef(BaseModel):
    name: ProviderHumanName
    module: ProviderName
    aliases: list[str]


class ProviderModelDef(BaseModel):
    provider_id: str
    model_id: str | None = None
    provider_def: ProviderDef


class ProviderModuleDef(BaseModel):
    provider_id: str
    entity_id: str | None = None
    provider_def: ProviderDef


BackendProviders = {
    "Ollama": ProviderDef(name="Ollama", module="ollama", aliases=[]),
    "OpenAI": ProviderDef(name="OpenAI", module="openai", aliases=["openai"]),
    "watsonx": ProviderDef(name="Watsonx", module="watsonx", aliases=["watsonx", "ibm"]),
    "Groq": ProviderDef(name="Groq", module="groq", aliases=["groq"]),
    "xAI": ProviderDef(name="XAI", module="xai", aliases=["xai", "grok"]),
    "vertexAI": ProviderDef(name="VertexAI", module="vertexai", aliases=["vertexai", "google"]),
    "Gemini": ProviderDef(name="Gemini", module="gemini", aliases=["gemini"]),
    "AmazonBedrock": ProviderDef(
        name="AmazonBedrock",
        module="amazon_bedrock",
        aliases=["amazon_bedrock", "amazon", "bedrock"],
    ),
    "anthropic": ProviderDef(name="Anthropic", module="anthropic", aliases=["anthropic"]),
    "AzureOpenAI": ProviderDef(
        name="AzureOpenAI",
        module="azure_openai",
        aliases=["azure_openai", "azure"],
    ),
    "mistralAI": ProviderDef(name="MistralAI", module="mistralai", aliases=["mistral"]),
    "Langchain": ProviderDef(name="LangChain", module="langchain", aliases=["langchain", "LangChain"]),
    "Llamaindex": ProviderDef(name="LlamaIndex", module="llamaindex", aliases=["llamaindex", "LlamaIndex"]),
    "BeeAI": ProviderDef(name="BeeAI", module="beeai", aliases=["BeeAI", "Beeai", "BAI"]),
}
