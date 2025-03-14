# BeeAI Framework Examples

This repository contains examples demonstrating the usage of the BeeAI Framework, a toolkit for building AI agents and applications.

## Table of Contents

1. [Agents](#agents)
2. [Workflows](#workflows)
3. [Cache](#cache)
4. [Errors](#errors)
5. [Helpers](#helpers)
6. [LLMs (Language Models)](#llms-language-models)
7. [Logger](#logger)
8. [Memory](#memory)
9. [Serialization](#serialization)
10. [Templates](#templates)
11. [Tools](#tools)

## Agents

- [`react.ts`](/typescript/examples/agents/react.ts): Basic ReAct Agent implementation
- [`react_advanced.ts`](/typescript/examples/agents/react_advanced.ts): Advanced ReAct Agent with custom configurations
- [`react_reusable.ts`](/typescript/examples/agents/react_reusable.ts): Demonstration of serializing and reusing ReAct Agents
- [`custom_agent.ts`](/typescript/examples/agents/custom_agent.ts): Example of creating a custom agent
- [`granite_react.ts`](/typescript/examples/agents/granite/granite_react.ts): Basic ReAct Agent using an IBM Granite LLM
- [`granite_wiki_react.ts`](/typescript/examples/agents/granite/granite_wiki_react.ts): Advanced ReAct Agent using an IBM Granite LLM with wikipedia retrieval
- [`simple.ts`](/typescript/examples/agents/simple.ts): Simple agent implementation
- [`sql.ts`](/typescript/examples/agents/sql.ts): Agent for SQL-related tasks

## Workflows

- [`simple.ts`](/typescript/examples/workflows/simple.ts): Introduction to workflows
- [`nesting.ts`](/typescript/examples/workflows/nesting.ts): How to nest workflows
- [`agent.ts`](/typescript/examples/workflows/agent.ts): Using workflows to interconnect two agents with a critique step.
- [`multiAgents.ts`](/typescript/examples/workflows/multiAgents.ts): Multi-step sequential agentic workflow.
- [`contentCreator.ts`](/typescript/examples/workflows/contentCreator.ts): Multi-step workflow for writing blog posts.

## Cache

- [`cacheFn.ts`](/typescript/examples/cache/cacheFn.ts): Function caching example
- [`custom.ts`](/typescript/examples/cache/custom.ts): Custom cache implementation
- [`decoratorCache.ts`](/typescript/examples/cache/decoratorCache.ts): Cache decorator usage
- [`decoratorCacheComplex.ts`](/typescript/examples/cache/decoratorCacheComplex.ts): Complex cache decorator example
- [`fileCache.ts`](/typescript/examples/cache/fileCache.ts): File-based caching
- [`fileCacheCustomProvider.ts`](/typescript/examples/cache/fileCacheCustomProvider.ts): Custom provider for file cache
- [`slidingCache.ts`](/typescript/examples/cache/slidingCache.ts): Sliding window cache implementation
- [`toolCache.ts`](/typescript/examples/cache/toolCache.ts): Caching for tools
- [`unconstrainedCache.ts`](/typescript/examples/cache/unconstrainedCache.ts): Unconstrained cache example
- [`unconstrainedCacheFunction.ts`](/typescript/examples/cache/unconstrainedCacheFunction.ts): Function using unconstrained cache

## Errors

- [`base.ts`](/typescript/examples/errors/base.ts): Basic error handling
- [`cast.ts`](/typescript/examples/errors/cast.ts): Error casting example
- [`tool.ts`](/typescript/examples/errors/tool.ts): Tool-specific error handling

## Helpers

- [`io.ts`](/typescript/examples/helpers/io.ts): Input/Output helpers
- [`setup.ts`](/typescript/examples/helpers/setup.ts): Setup utilities

## LLMs (Language Models)

- [`chat.ts`](/typescript/examples/backend/chat.ts): Chat-based language model usage
- [`chatCallback.ts`](/typescript/examples/backend/chatStream.ts): Callbacks for chat models
- [`structured.ts`](/typescript/examples/backend/structured.ts): Structured output from language models

### LLM Providers

- [`groq.ts`](/typescript/examples/backend/providers/groq.ts): Groq language model integration
- [`langchain.ts`](/typescript/examples/backend/providers/langchain.ts): LangChain integration
- [`ollama.ts`](/typescript/examples/backend/providers/ollama.ts): Ollama model usage
- [`openai.ts`](/typescript/examples/backend/providers/openai.ts): OpenAI integration
- [`watsonx.ts`](/typescript/examples/backend/providers/watsonx.ts): Watsonx integration
- [`anthropic.ts`](/typescript/examples/backend/providers/anthropic.ts): Anthropic integration

## Logger

- [`agent.ts`](/typescript/examples/logger/agent.ts): Agent-specific logging
- [`base.ts`](/typescript/examples/logger/base.ts): Basic logging setup
- [`pino.ts`](/typescript/examples/logger/pino.ts): Pino logger integration

## Memory

- [`agentMemory.ts`](/typescript/examples/memory/agentMemory.ts): Memory management for agents
- [`custom.ts`](/typescript/examples/memory/custom.ts): Custom memory implementation
- [`llmMemory.ts`](/typescript/examples/memory/llmMemory.ts): Memory for language models
- [`slidingMemory.ts`](/typescript/examples/memory/slidingMemory.ts): Sliding window memory
- [`summarizeMemory.ts`](/typescript/examples/memory/summarizeMemory.ts): Memory with summarization
- [`tokenMemory.ts`](/typescript/examples/memory/tokenMemory.ts): Token-based memory
- [`unconstrainedMemory.ts`](/typescript/examples/memory/unconstrainedMemory.ts): Unconstrained memory example

## Serialization

- [`base.ts`](/typescript/examples/serialization/base.ts): Basic serialization
- [`context.ts`](/typescript/examples/serialization/context.ts): Context serialization
- [`customExternal.ts`](/typescript/examples/serialization/customExternal.ts): Custom external serialization
- [`customInternal.ts`](/typescript/examples/serialization/customInternal.ts): Custom internal serialization
- [`memory.ts`](/typescript/examples/serialization/memory.ts): Memory serialization

## Templates

- [`arrays.ts`](/typescript/examples/templates/arrays.ts): Array-based templates
- [`forking.ts`](/typescript/examples/templates/forking.ts): Template forking
- [`functions.ts`](/typescript/examples/templates/functions.ts): Function-based templates
- [`objects.ts`](/typescript/examples/templates/objects.ts): Object-based templates
- [`primitives.ts`](/typescript/examples/templates/primitives.ts): Primitive data type templates

## Tools

- [`advanced.ts`](/typescript/examples/tools/advanced.ts): Advanced tool usage
- [`agent.ts`](/typescript/examples/tools/agent.ts): Agent-specific tools
- [`base.ts`](/typescript/examples/tools/base.ts): Basic tool implementation
- [`mcp.ts`](/typescript/examples/tools/mcp.ts): MCP tool usage

### Custom Tools

- [`base.ts`](/typescript/examples/tools/custom/base.ts): Custom tool base implementation
- [`dynamic.ts`](/typescript/examples/tools/custom/dynamic.ts): Dynamic tool creation
- [`openLibrary.ts`](/typescript/examples/tools/custom/openLibrary.ts): OpenLibrary API tool
- [`python.ts`](/typescript/examples/tools/custom/python.ts): Python-based custom tool

- [`langchain.ts`](/typescript/examples/tools/langchain.ts): LangChain tool integration

## Usage

To run these examples, make sure you have the BeeAI Framework cloned and properly configured. Each file demonstrates a specific feature or use case of the framework. You can run individual examples using Node.js with TypeScript support.

1. Clone the repository:
   ```shell
   git clone git@github.com:i-am-bee/beeai-framework
   ```
2. Install dependencies:
   ```shell
   yarn install --immutable && yarn prepare
   ```
3. Create `.env` file (from `.env.template`) and fill in missing values (if any).

4. Run an arbitrary example, use the following command:

   ```shell
   yarn start examples/path/to/example.ts
   ```

For more detailed information on the BeeAI Framework, please refer to the [documentation](/docs/README.md).

> [!TIP]
>
> To run examples that use Ollama, be sure that you have installed [Ollama](https://ollama.com) with the [llama3.1](https://ollama.com/library/llama3.1) model downloaded.
