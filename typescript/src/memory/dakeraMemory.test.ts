/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { verifyDeserialization } from "@tests/e2e/utils.js";
import { Message } from "@/backend/message.js";
import { DakeraMemory } from "@/memory/dakeraMemory.js";

// ---------------------------------------------------------------------------
// Mock global fetch so no live server is needed
// ---------------------------------------------------------------------------

const STORE_RESPONSE = {
  memory: {
    id: "mem-001",
    content: "hello from dakera",
    agent_id: "test-agent",
    importance: 0.5,
    tags: ["test-agent", "bee"],
    metadata: {},
  },
};

const RECALL_RESPONSE = {
  memories: [
    {
      memory: {
        id: "mem-001",
        content: "hello from dakera",
        agent_id: "test-agent",
        importance: 0.5,
        tags: [],
        metadata: {},
      },
      score: 0.92,
    },
  ],
};

function mockFetch(response: object, status = 200) {
  return vi.fn().mockResolvedValue({
    ok: status >= 200 && status < 300,
    status,
    statusText: status === 200 ? "OK" : "Error",
    text: async () => JSON.stringify(response),
    json: async () => response,
  } as Response);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("DakeraMemory", () => {
  let originalFetch: typeof global.fetch;

  beforeEach(() => {
    originalFetch = global.fetch;
  });

  afterEach(() => {
    global.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  it("initialises with empty in-session message list", () => {
    const memory = new DakeraMemory({ url: "http://localhost:3000", agentId: "agent1" });
    expect(memory.messages).toHaveLength(0);
  });

  it("add() stores the message in Dakera and appends it locally", async () => {
    global.fetch = mockFetch(STORE_RESPONSE) as unknown as typeof global.fetch;
    const memory = new DakeraMemory({
      url: "http://localhost:3000",
      agentId: "agent1",
      topK: 0, // disable recall injection for this test
    });
    const msg = Message.of({ role: "user", text: "hello from dakera" });
    await memory.add(msg);

    expect(global.fetch).toHaveBeenCalledTimes(1);
    const [url, opts] = vi.mocked(global.fetch).mock.calls[0] as [string, RequestInit];
    expect(url).toBe("http://localhost:3000/v1/memory/store");
    const body = JSON.parse(opts.body as string);
    expect(body.agent_id).toBe("agent1");
    expect(body.content).toBe("hello from dakera");

    expect(memory.messages).toHaveLength(1);
    expect(memory.messages[0].text).toBe("hello from dakera");
  });

  it("add() recalls memories and injects them as system messages when topK > 0", async () => {
    // First call: store, second call: recall
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        text: async () => JSON.stringify(STORE_RESPONSE),
        json: async () => STORE_RESPONSE,
      })
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        text: async () => JSON.stringify(RECALL_RESPONSE),
        json: async () => RECALL_RESPONSE,
      });
    global.fetch = fetchMock as unknown as typeof global.fetch;

    const memory = new DakeraMemory({
      url: "http://localhost:3000",
      agentId: "agent1",
      topK: 5,
      persist: true,
    });
    const msg = Message.of({ role: "user", text: "hello" });
    await memory.add(msg);

    // Store then recall — the recall endpoint is /v1/memory/recall.
    expect(fetchMock).toHaveBeenCalledTimes(2);
    const [recallUrl] = fetchMock.mock.calls[1] as [string, RequestInit];
    expect(recallUrl).toBe("http://localhost:3000/v1/memory/recall");

    // Should have the original message plus a recalled system message.
    const systemMsgs = memory.messages.filter((m) => m.role === "system");
    expect(systemMsgs.length).toBeGreaterThan(0);
  });

  it("add() with persist=false skips the store call but still recalls memories", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      text: async () => JSON.stringify(RECALL_RESPONSE),
      json: async () => RECALL_RESPONSE,
    });
    global.fetch = fetchMock as unknown as typeof global.fetch;

    const memory = new DakeraMemory({
      url: "http://localhost:3000",
      agentId: "agent1",
      persist: false, // read-only: no writes, but still recalls
      topK: 5,
    });
    const msg = Message.of({ role: "user", text: "read-only test" });
    await memory.add(msg);

    // With persist=false, only the recall call is made (no store call at all).
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("http://localhost:3000/v1/memory/recall");
  });

  it("delete() removes the message from the local list", async () => {
    global.fetch = mockFetch(STORE_RESPONSE) as unknown as typeof global.fetch;
    const memory = new DakeraMemory({
      url: "http://localhost:3000",
      agentId: "agent1",
      topK: 0,
    });
    const msg = Message.of({ role: "user", text: "delete me" });
    await memory.add(msg);
    expect(memory.messages).toHaveLength(1);
    const deleted = await memory.delete(msg);
    expect(deleted).toBe(true);
    expect(memory.messages).toHaveLength(0);
  });

  it("reset() clears the local message list", async () => {
    global.fetch = mockFetch(STORE_RESPONSE) as unknown as typeof global.fetch;
    const memory = new DakeraMemory({
      url: "http://localhost:3000",
      agentId: "agent1",
      topK: 0,
    });
    await memory.add(Message.of({ role: "user", text: "msg1" }));
    await memory.add(Message.of({ role: "user", text: "msg2" }));
    expect(memory.messages).toHaveLength(2);
    memory.reset();
    expect(memory.messages).toHaveLength(0);
  });

  it("does not crash when the Dakera store call fails", async () => {
    global.fetch = mockFetch({ error: "server error" }, 500) as unknown as typeof global.fetch;
    const memory = new DakeraMemory({
      url: "http://localhost:3000",
      agentId: "agent1",
      topK: 0,
    });
    const msg = Message.of({ role: "user", text: "resilient" });
    // Should not throw — a Dakera failure is non-fatal.
    await expect(memory.add(msg)).resolves.not.toThrow();
  });

  it("sends the Authorization header when apiKey is set", async () => {
    global.fetch = mockFetch(STORE_RESPONSE) as unknown as typeof global.fetch;
    const memory = new DakeraMemory({
      url: "http://localhost:3000",
      apiKey: "dk-secret-key",
      agentId: "agent1",
      topK: 0,
    });
    const msg = Message.of({ role: "user", text: "auth test" });
    await memory.add(msg);

    const [, opts] = vi.mocked(global.fetch).mock.calls[0] as [string, RequestInit];
    const headers = opts.headers as Record<string, string>;
    expect(headers["Authorization"]).toBe("Bearer dk-secret-key");
  });

  it("never serializes the apiKey", async () => {
    const memory = new DakeraMemory({
      url: "http://localhost:3000",
      agentId: "agent1",
      apiKey: "dk-super-secret",
    });

    // The snapshot must not carry the secret.
    const snapshot = memory.createSnapshot();
    expect(snapshot).not.toHaveProperty("apiKey");

    // Neither may the fully serialized checkpoint string.
    const serialized = await memory.serialize();
    expect(serialized).not.toContain("dk-super-secret");
  });

  it("round-trips through serialize / fromSerialized", async () => {
    // No apiKey — it is intentionally dropped on deserialization, so a round-trip
    // is only structurally identical when the key is absent to begin with.
    const memory = new DakeraMemory({
      url: "http://localhost:3000",
      agentId: "agent1",
      sessionId: "sess-1",
      topK: 0, // avoid any network calls in add()
      persist: false,
    });
    await memory.add(Message.of({ role: "user", text: "remember me" }));

    const serialized = await memory.serialize();
    const deserialized = await DakeraMemory.fromSerialized(serialized);
    verifyDeserialization(memory, deserialized);
  });
});
