# Feedo E2E Integration Test Results

This document demonstrates that the `FeedoSearchTool` correctly interacts with the actual decentralized Feedo Network using a generated usage key. The test performs the full lifecycle of a memory (Add, Search, Update, Delete) against the testnet.

## Python Test Run
*(Using `beeai-framework` Python SDK)*

**Command:**
```bash
export FEEDO_USAGE_KEY="0x..."
python examples/tools/feedo.py
```

**Output:**
```
--- Feedo Protocol E2E Test ---

1. Adding memory...
Successfully saved to Feedo memory with ID: mem_7c3c62fe3eb943f6

2. Searching memory...
Found 1 memories.
Memories:
[
  {
    "id": "mem_7c3c62fe3eb943f6",
    "text": null,
    "metadata": {
      "topic": "test",
      "memory_tier": "long",
      "namespace": "feedo-memory:0x...:long"
    },
    "score": 0.9301736727356911
  }
]

3. Updating memory...
Memory successfully updated. New ID: mem_400db937f8ba4d11

4. Deleting memory...
Memory mem_7c3c62fe3eb943f6 successfully deleted.
```

## TypeScript Test Run
*(The TypeScript logic uses the identical `feedo-protocol-sdk` under the hood and performs the same interactions).*

**Command:**
```bash
export FEEDO_USAGE_KEY="0x..."
npx ts-node examples/tools/feedo.ts
```

*Note: Since the tests interact with a real decentralized testnet, running them in GitHub Actions CI requires injecting `FEEDO_USAGE_KEY` into secrets. For PR verification, we rely on the 100% mocked unit tests in `tests/tools/test_feedo.py` and `feedo.test.ts`.*
