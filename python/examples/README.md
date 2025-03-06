# BeeAI Framework Examples

This repository contains examples demonstrating the usage of the BeeAI Framework, a toolkit for building AI agents and applications.

## Table of Contents

1. [Agents](#agents)
2. [Workflows](#workflows)
3. [Helpers](#helpers)
4. [Backend](#backend)
5. [Memory](#memory)
6. [Serialization](#serialization)
7. [Templates](#templates)
8. [Tools](#tools)

## Agents

- [`react.py`](/python/examples/agents/react.py): Basic ReAct Agent implementation
- [`simple.py`](/python/examples/agents/simple.py): Simple agent implementation

## Workflows

- [`simple.py`](/python/examples/workflows/simple.py): Introduction to workflows
- [`multi_agents.py`](/python/examples/workflows/multi_agents.py): Multi-step sequential agentic workflow.
- [`web_agent.py`](/python/examples/workflows/web_agent.py): Web Agent

## Helpers

- [`io.py`](/python/examples/helpers/io.py): Input/Output helpers

## Backend

- [`ollama.py`](/python/examples/backend/providers/ollama.py): Ollama Chat-based language model usage
- [`watsonx.py`](/python/examples/backend/providers/watsonx.py): Watsonx Chat-based language model usage

### LLM Providers

- [`ollama.py`](/python/examples/backend/providers/ollama.py): Ollama model usage
- [`watsonx.py`](/python/examples/backend/providers/watsonx.py): Watsonx integration

## Memory

- [unconstrained_memory.py](/examples/memory/unconstrained_memory.py) - Basic memory usage
- [sliding_memory.py](/examples/memory/sliding_memory.py) - Sliding window memory example
- [token_memory.py](/examples/memory/token_memory.py) - Token-based memory management
- [summarize_memory.py](/examples/memory/summarize_memory.py) - Summarization memory example
- [agent_memory.py](/examples/memory/agent_memory.py) - Using memory with agents

## Templates

- [`basic_template.py`](/python/examples/templates/basic_template.py): Simple template
- [`functions.py`](/python/examples/templates/functions.py): Template functions
- [`objects.py`](/python/examples/templates/objects.py): Working with objects
- [`arrays.py`](/python/examples/templates/arrays.py): Working with arrays
- [`forking.py`](/python/examples/templates/forking.py): Template forking
- [`system_prompt.py`](/python/examples/templates/system_prompt.py): Using templates with agents

## Tools

- [`decorator.py`](/python/examples/tools/decorator.py): Tool creation using decorator
- [`duckduckgo.py`](/python/examples/tools/duckduckgo.py): DDG Search Tool
- [`openmeteo.py`](/python/examples/tools/openmeteo.py): DDG Search Tool
