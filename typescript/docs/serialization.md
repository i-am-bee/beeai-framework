# Serialization

> [!TIP]
>
> Location within the framework `beeai-framework/serializer`.

Serialization is a process of converting complex data structures or objects into a format that can be easily stored, transmitted, and reconstructed later.
Serialization is a difficult task, and JavaScript does not provide a magic tool to serialize and deserialize an arbitrary input. That is why we made such one.

<!-- embedme examples/serialization/base.ts -->

```ts
import { Serializer } from "beeai-framework/serializer/serializer";

const original = new Date("2024-01-01T00:00:00.000Z");
const serialized = await Serializer.serialize(original);
const deserialized = await Serializer.deserialize(serialized);

console.info(deserialized instanceof Date); // true
console.info(original.toISOString() === deserialized.toISOString()); // true
```

_Source: [examples/serialization/base.ts](/typescript/examples/tools/base.ts)_

> [!NOTE]
>
> Serializer knows how to serialize/deserialize the most well-known JavaScript data structures. Continue reading to see how to register your own.

## Being Serializable

Most parts of the framework implement the internal [`Serializable`](/typescript/src/internals/serializable.ts) class, which exposes the following methods.

- `createSnapshot` (returns an object that "snapshots" the current state)
- `loadSnapshot` (applies the provided snapshot to the current instance)

- `fromSerialized` (static, creates the new instance from the given serialized input)
- `fromSnapshot` (static, creates the new instance from the given snapshot)

See the direct usage on the following memory example.

<!-- embedme examples/serialization/memory.ts -->

```ts
import { TokenMemory } from "beeai-framework/memory/tokenMemory";
import { AssistantMessage, UserMessage } from "beeai-framework/backend/message";

const memory = new TokenMemory();
await memory.add(new UserMessage("What is your name?"));

const serialized = await memory.serialize();
const deserialized = await TokenMemory.fromSerialized(serialized);

await deserialized.add(new AssistantMessage("Bee"));
```

_Source: [examples/serialization/memory.ts](/typescript/examples/serialization/memory.ts)_

### Serializing unknowns

If you want to serialize a class that the `Serializer` does not know, it throws the `SerializerError` error.
However, you can tell the `Serializer` how to work with your class by registering it as a serializable.

<!-- embedme examples/serialization/customExternal.ts -->

```ts
import { Serializer } from "beeai-framework/serializer/serializer";

class MyClass {
  constructor(public readonly name: string) {}
}

Serializer.register(MyClass, {
  // Defines how to transform a class to a plain object (snapshot)
  toPlain: (instance) => ({ name: instance.name }),
  // Defines how to transform a plain object (snapshot) a class instance
  fromPlain: (snapshot) => new MyClass(snapshot.name),

  // optional handlers to support lazy initiation (handling circular dependencies)
  createEmpty: () => new MyClass(""),
  updateInstance: (instance, update) => {
    Object.assign(instance, update);
  },
});

const instance = new MyClass("Bee");
const serialized = await Serializer.serialize(instance);
const deserialized = await Serializer.deserialize<MyClass>(serialized);

console.info(instance);
console.info(deserialized);
```

_Source: [examples/serialization/customExternal.ts](/typescript/examples/serialization/customExternal.ts)_

or you can extend the `Serializable` class.

<!-- embedme examples/serialization/customInternal.ts -->

```ts
import { Serializable } from "beeai-framework/internals/serializable";

class MyClass extends Serializable {
  constructor(public readonly name: string) {
    super();
  }

  static {
    // register class to the global serializer register
    this.register();
  }

  createSnapshot(): unknown {
    return {
      name: this.name,
    };
  }

  loadSnapshot(snapshot: ReturnType<typeof this.createSnapshot>) {
    Object.assign(this, snapshot);
  }
}

const instance = new MyClass("Bee");
const serialized = await instance.serialize();
const deserialized = await MyClass.fromSerialized(serialized);

console.info(instance);
console.info(deserialized);
```

_Source: [examples/serialization/customInternal.ts](/typescript/examples/serialization/customInternal.ts)_

> [!TIP]
>
> Most framework components are `Serializable`.

### Context matters

<!-- embedme examples/serialization/context.ts -->

```ts
import { UnconstrainedMemory } from "beeai-framework/memory/unconstrainedMemory";
import { UserMessage } from "beeai-framework/backend/message";

// String containing serialized `UnconstrainedMemory` instance with one message in it.
const serialized = `{"__version":"0.0.0","__root":{"__serializer":true,"__class":"Object","__ref":"18","__value":{"target":"UnconstrainedMemory","snapshot":{"__serializer":true,"__class":"Object","__ref":"17","__value":{"messages":{"__serializer":true,"__class":"Array","__ref":"1","__value":[{"__serializer":true,"__class":"SystemMessage","__ref":"2","__value":{"content":{"__serializer":true,"__class":"Array","__ref":"3","__value":[{"__serializer":true,"__class":"Object","__ref":"4","__value":{"type":"text","text":"You are a helpful assistant."}}]},"meta":{"__serializer":true,"__class":"Object","__ref":"5","__value":{"createdAt":{"__serializer":true,"__class":"Date","__ref":"6","__value":"2025-02-06T14:51:01.459Z"}}},"role":"system"}},{"__serializer":true,"__class":"UserMessage","__ref":"7","__value":{"content":{"__serializer":true,"__class":"Array","__ref":"8","__value":[{"__serializer":true,"__class":"Object","__ref":"9","__value":{"type":"text","text":"Hello!"}}]},"meta":{"__serializer":true,"__class":"Object","__ref":"10","__value":{"createdAt":{"__serializer":true,"__class":"Date","__ref":"11","__value":"2025-02-06T14:51:01.459Z"}}},"role":"user"}},{"__serializer":true,"__class":"AssistantMessage","__ref":"12","__value":{"content":{"__serializer":true,"__class":"Array","__ref":"13","__value":[{"__serializer":true,"__class":"Object","__ref":"14","__value":{"type":"text","text":"Hello, how can I help you?"}}]},"meta":{"__serializer":true,"__class":"Object","__ref":"15","__value":{"createdAt":{"__serializer":true,"__class":"Date","__ref":"16","__value":"2025-02-06T14:51:01.459Z"}}},"role":"assistant"}}]}}}}}}`;

// If `Message` was not imported the serialization would fail because the `Message` had no chance to register itself.
const memory = await UnconstrainedMemory.fromSerialized(serialized, {
  // this part can be omitted if all classes used in the serialized string are imported (and have `static` register block) or at least one initiated
  extraClasses: [UserMessage],
});
console.info(memory.messages);
```

_Source: [examples/serialization/context.ts](/typescript/examples/serialization/context.ts)_

> [!IMPORTANT]
>
> Ensuring that all classes are registered in advance can be annoying, but there's a good reason for that.
> If we imported all the classes for you, that would significantly increase your application's size and bootstrapping time + you would have to install all peer dependencies that you may not even need.
