# Emitter (Observability)

*Disclaimer: The notes below may refer to the TypeScript version or missing files as the Python version moves toward parity in the near future. Additional Python examples coming soon. TODO*

> Location within the framework `python/beeai_framework/emitter`.

An emitter is a core functionality of the framework that allows you to see what is happening under the hood.

## Standalone usage

The following examples demonstrate how [`Emitter`](/beeai/utils/events.py) concept works.

### Basic Usage

```py
```

_Source: /examples/emitter/base.py TODO

> [!NOTE]
>
> You can create your own emitter by initiating the `Emitter` class, but typically it's better to use or fork the root one.

### Advanced

```py
```

_Source: /examples/emitter/advanced.py TODO


### Event Matching

```py
```

_Source: /examples/emitter/matchers.py TODO

### Event Piping


```py
```

_Source: /examples/emitter/piping.py TODO

## Framework Usage

Typically, you consume out-of-the-box modules that use the `Emitter` concept on your behalf.

## Agent usage


```py
```

_Source: /examples/emitter/agentMatchers.py TODO
