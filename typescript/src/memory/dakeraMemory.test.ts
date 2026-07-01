/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { jest } from "@jest/globals";
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

const SEARCH_RESPONSE = {
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
  return jest.fn().mockResolvedValue({
    ok: status >= 200 && status < 300,
    status,
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
  });

  it("initialises with empty in-session message list", () => {
    const memory = new DakeraMemory({ url: "http://localhost:3300", agentId: "agent1" });
    expect(memory.messages).toHaveLength(0);
  });

  it("add() stores the message in Dakera and appends it locally", async () => {
    global.fetch = mockFetch(STORE_RESPONSE) as unknown as typeof global.fetch;
    const memory = new DakeraMemory({
      url: "http://localhost:3300",
      agentId: "agent1",
      topK: 0, // disable recall injection for this test
    });
    const msg = Message.of({ role: "user", text: "hello from dakera" });
    await memory.add(msg);

    expect(global.fetch).toHaveBeenCalledTimes(1);
    const [url, opts] = (global.fetch as jest.Mock).mock.calls[0] as [string, RequestInit];
    expect(url).toBe("http://localhost:3300/v1/memory/store");
    const body = JSON.parse(opts.body as string);
    expect(body.agent_id).toBe("agent1");
    expect(body.content).toBe("hello from dakera");

    expect(memory.messages).toHaveLength(1);
    expect(memory.messages[0].text).toBe("hello from dakera");
  });

  it("add() injects recalled memories as system messages when topK > 0", async () => {
    // First call: store, second call: search
    const fetchMock = jest
      .fn()
      .mockResolvedValueOnce({ ok: true, status: 200, json: async () => STORE_RESPONSE })
      .mockResolvedValueOnce({ ok: true, status: 200, json: async () => SEARCH_RESPONSE });
    global.fetch = fetchMock as unknown as typeof global.fetch;

    const memory = new DakeraMemory({
      url: "http://localhost:3300",
      agentId: "agent1",
      topK: 5,
      persist: true,
    });
    const msg = Message.of({ role: "user", text: "hello" });
    await memory.add(msg);

    // Should have the original message plus a recalled system message
    const systemMsgs = memory.messages.filter((m) => m.role === "system");
    expect(systemMsgs.length).toBeGreaterThan(0);
  });

  it("add() with persist=false skips the store call but still injects recalled memories", async () => {
    const fetchMock = jest
      .fn()
      .mockResolvedValue({ ok: true, status: 200, json: async () => SEARCH_RESPONSE });
    global.fetch = fetchMock as unknown as typeof global.fetch;

    const memory = new DakeraMemory({
      url: "http://localhost:3300",
      agentId: "agent1",
      persist: false,  // read-only: no writes, but still recalls
      topK: 5,
    });
    const msg = Message.of({ role: "user", text: "read-only test" });
    await memory.add(msg);

    // With persist=false, only the search call is made (no store call at all)
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url] = (fetchMock as jest.Mock).mock.calls[0] as [string, RequestInit];
    expect(url).toBe("http://localhost:3300/v1/memory/search");
  });

  it("delete() removes the message from the local list", async () => {
    global.fetch = mockFetch(STORE_RESPONSE) as unknown as typeof global.fetch;
    const memory = new DakeraMemory({
      url: "http://localhost:3300",
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
      url: "http://localhost:3300",
      agentId: "agent1",
      topK: 0,
    });
    await memory.add(Message.of({ role: "user", text: "msg1" }));
    await memory.add(Message.of({ role: "user", text: "msg2" }));
    expect(memory.messages).toHaveLength(2);
    memory.reset();
    expect(memory.messages).toHaveLength(0);
  });

  it("does not crash when Dakera store call fails", async () => {
    global.fetch = mockFetch({ error: "server error" }, 500) as unknown as typeof global.fetch;
    const memory = new DakeraMemory({
      url: "http://localhost:3300",
      agentId: "agent1",
      topK: 0,
    });
    const msg = Message.of({ role: "user", text: "resilient" });
    // Should not throw — Dakera failure is non-fatal
    await expect(memory.add(msg)).resolves.not.toThrow();
  });

  it("sends Authorization header when apiKey is set", async () => {
    global.fetch = mockFetch(STORE_RESPONSE) as unknown as typeof global.fetch;
    const memory = new DakeraMemory({
      url: "http://localhost:3300",
      apiKey: "my-secret-key",
      agentId: "agent1",
      topK: 0,
    });
    const msg = Message.of({ role: "user", text: "auth test" });
    await memory.add(msg);

    const [, opts] = (global.fetch as jest.Mock).mock.calls[0] as [string, RequestInit];
    const headers = opts.headers as Record<string, string>;
    expect(headers["Authorization"]).toBe("Bearer my-secret-key");
  });

  it("snapshot/loadSnapshot round-trips correctly", async () => {
    const memory = new DakeraMemory({
      url: "http://localhost:3300",
      agentId: "agent1",
      apiKey: "test-key",
      sessionId: "sess-1",
      topK: 3,
      persist: false,
    });
    const snapshot = memory.createSnapshot();
    expect(snapshot).toHaveProperty("url", "http://localhost:3300");
    expect(snapshot).toHaveProperty("agentId", "agent1");
    expect(snapshot).toHaveProperty("topK", 3);
    expect(snapshot).toHaveProperty("persist", false);

    // loadSnapshot restores all fields — use valid required fields for the new instance
    const memory2 = new DakeraMemory({ url: "http://placeholder:3300", agentId: "placeholder" });
    memory2.loadSnapshot(snapshot);
    expect(memory2.createSnapshot()).toEqual(snapshot);
  });
});
