# 📊 OpenTelemetry Instrumentation

> [!NOTE]  
> **COMING SOON! 🚀 Instrumentation is not yet implemented in Python, but is available today in [TypeScript](https://github.com/Arize-ai/openinference/tree/main/js/packages/openinference-instrumentation-beeai)**

This document provides an overview of the OpenTelemetry instrumentation setup in beeai-framework.
The implementation is designed to [create telemetry spans](https://opentelemetry.io/docs/languages/python/instrumentation/#creating-spans) for observability when instrumentation is enabled.

## Overview

OpenTelemetry instrumentation allows you to collect telemetry data, such as traces and metrics, to monitor the performance of your services.
This setup involves creating middleware to handle instrumentation automatically when the `INSTRUMENTATION_ENABLED` flag is active.

## Setting up OpenTelemetry

Follow the official OpenTelemetry [Python Getting Started Guide](https://opentelemetry.io/docs/languages/python/getting-started/) to initialize and configure OpenTelemetry in your application.

## Instrumentation Configuration

### Environment Variable

Use the environment variable `BEE_FRAMEWORK_INSTRUMENTATION_ENABLED` to enable or disable instrumentation.

```bash
# Enable instrumentation
export BEE_FRAMEWORK_INSTRUMENTATION_ENABLED=true
# Ignore sensitive keys from collected events data
export INSTRUMENTATION_IGNORED_KEYS="apiToken,accessToken"
```

If `BEE_FRAMEWORK_INSTRUMENTATION_ENABLED` is false or unset, the framework will run without instrumentation.

## Creating Custom Spans

You can manually create spans during the `run` process to track specific parts of the execution. This is useful for adding custom telemetry to enhance observability.

Example of creating a span:

```txt
Coming soon
```

## Verifying Instrumentation

Once you have enabled the instrumentation, you can view telemetry data using any [compatible OpenTelemetry backend](https://opentelemetry.io/docs/languages/js/exporters/), such as [Jaeger](https://www.jaegertracing.io/), [Zipkin](https://zipkin.io/), [Prometheus](https://prometheus.io/docs/prometheus/latest/feature_flags/#otlp-receiver), etc...
Ensure your OpenTelemetry setup is properly configured to export trace data to your chosen backend.

### Agent instrumentation

Running the Instrumented Application file.

```bash
TODO
```

### LLM instrumentation

Running the LLM Instrumentation file.

```bash
TODO
```

### Tool instrumentation

Running Tool Instrumentation file.

```bash
TODO
```

## Conclusion

This setup provides basic OpenTelemetry instrumentation with the flexibility to enable or disable it as needed.
By creating custom spans and using `createTelemetryMiddleware`, you can capture detailed telemetry for better observability and performance insights.
