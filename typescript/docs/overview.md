# Overview

## 📦 Modules

The source directory (`src`) provides numerous modules that one can use.

| Name                                 | Description                                                                                 |
| ------------------------------------ | ------------------------------------------------------------------------------------------- |
| [**agents**](./agents.md)            | Base classes defining the common interface for agent.                                       |
| [**backend**](./backend.md)          | Functionalities that relates to AI models (chat, embedding, image, tool calling, ...)       |
| [**template**](./templates.md)       | Prompt Templating system based on `Mustache` with various improvements.                     |
| [**memory**](./memory.md)            | Various types of memories to use with agent.                                                |
| [**tools**](./tools.md)              | Tools that an agent can use.                                                                |
| [**cache**](./cache.md)              | Preset of different caching approaches that can be used together with tools.                |
| [**errors**](./errors.md)            | Base framework error classes used by each module.                                           |
| [**logger**](./logger.md)            | Core component for logging all actions within the framework.                                |
| [**serializer**](./serialization.md) | Core component for the ability to serialize/deserialize modules into the serialized format. |
| [**version**](./version.md)          | Constants representing the framework (e.g., the latest version)                             |
| [**emitter**](./emitter.md)          | Bringing visibility to the system by emitting events.                                       |
| **internals**                        | Modules used by other modules within the framework.                                         |

### Emitter

Moved to a [standalone page](emitter.md).

### LLMs

Moved to a [standalone page](backend.md).

### Templates

Moved to a [standalone page](templates.md).

### Agents

Moved to a [standalone page](agents.md).

### Memory

Moved to a [standalone page](memory.md).

### Tools

Moved to a [standalone page](tools.md).

### Cache

Moved to a [standalone page](cache.md).

### Errors

Moved to a [standalone page](errors.md).

### Logger

Moved to a [standalone page](logger.md).

### Serializer

Moved to a [standalone page](serialization.md).
