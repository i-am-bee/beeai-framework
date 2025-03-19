# Amazon Bedrock

## Configuration

Set the following environment variables

* AWS_ACCESS_KEY_ID
* AWS_SECRET_ACCESS_KEY
* AWS_REGION

## Tested Models

Only Meta, Mistral, & Amazon Titan serverless models have been tested.

Other models should work, as beeai_framework uses LiteLLM. See [docs](https://docs.litellm.ai/docs/providers/bedrock) for more information

## Known Issues with tool use and structured output

The following models report Tool use not supported:

```text
litellm.llms.bedrock.common_utils.BedrockError: {"message":"This model doesn't support tool use."}
```

* `meta.llama3-70b-instruct-v1:0`
* `meta.llama3-8b-instruct-v1:0`

The following fail to return structured output with beeai_framework. Initial investigation indicates that these models are not responding with structured JSON output when requested

* `amazon.titan-text-express-v1`
* `amazon.titan-text-lite-v1`
* `mistral.mistral-7b-instruct-v0:2`
* `mistral.mixtral-8x7b-instruct-v0:1`
* `mistral.mistral-large-2402-v1:0`

The following models fail with an exception:

```text
litellm.exceptions.BadRequestError: litellm.BadRequestError: BedrockException - {"message":"This model doesn't support the toolConfig.toolChoice.tool field. Remove toolConfig.toolChoice.tool and try again."}
```

* `mistral.mistral-large-2402-v1:0`

## Quota limits

Default quota limits on Amazon Bedrock are low, and can cause even
simple examples to fail with:

```text
litellm.exceptions.RateLimitError: litellm.RateLimitError: BedrockException - {"message":"Too many requests, please wait before trying again."}
```

To increase quota limits, see [Amazon Bedrock pricing](https://aws.amazon.com/bedrock/pricing/) and
[Amazon Bedrock quotas](https://docs.aws.amazon.com/bedrock/latest/userguide/quotas.html).