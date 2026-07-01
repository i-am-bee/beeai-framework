/**
 * DakeraMemory — persistent, decay-weighted vector memory for BeeAI agents
 *
 * Integrates with the Dakera REST API (https://dakera.ai).
 * Self-hosted: `docker run -p 3300:3300 -e DAKERA_API_KEY=demo ghcr.io/dakera-ai/dakera:latest`
 *
 * Usage:
 *   const memory = new DakeraMemory({ url: "http://localhost:3300", agentId: "my-agent" });
 *   const agent = new ReActAgent({ llm, tools, memory });
 *
 * On every add(), the message is persisted to Dakera and the in-session window is enriched
 * with semantically relevant memories retrieved from past sessions.
 */

import { BaseMemory, MemoryError } from "@/memory/base.js";
import { Message } from "@/backend/message.js";

// ---------------------------------------------------------------------------
// Dakera API types
// ---------------------------------------------------------------------------

interface DakeraStoreRequest {
  content: string;
  agent_id: string;
  session_id?: string;
  importance?: number;
  tags?: string[];
  metadata?: Record<string, unknown>;
}

interface DakeraMemoryRecord {
  id: string;
  content: string;
  agent_id: string;
  session_id?: string;
  importance?: number;
  tags?: string[];
  metadata?: Record<string, unknown>;
  created_at?: string;
}

interface DakeraStoreResponse {
  memory: DakeraMemoryRecord;
}

interface DakeraSearchRequest {
  agent_id: string;
  query: string;
  top_k?: number;
  session_id?: string;
}

interface DakeraSearchHit {
  memory: DakeraMemoryRecord;
  score: number;
}

interface DakeraSearchResponse {
  memories: DakeraSearchHit[];
}

interface DakeraForgetRequest {
  agent_id: string;
  memory_ids?: string[];
  session_id?: string;
}

// ---------------------------------------------------------------------------
// Constructor input
// ---------------------------------------------------------------------------

export interface DakeraMemoryInput {
  /** Base URL of the Dakera server, e.g. "http://localhost:3300" */
  url: string;
  /** Optional API key — sent as `Authorization: Bearer <apiKey>` */
  apiKey?: string;
  /** Logical agent identifier stored with every memory record */
  agentId: string;
  /** Optional session identifier — scopes retrieval to a single session */
  sessionId?: string;
  /**
   * Number of past memories to inject above the current conversation window.
   * Defaults to 5. Set to 0 to disable injection.
   */
  topK?: number;
  /**
   * Whether to store every message that is added to this memory in Dakera.
   * Defaults to true. Set to false to make the instance read-only against Dakera
   * (messages are still stored in the local window).
   */
  persist?: boolean;
}

// ---------------------------------------------------------------------------
// Snapshot shape (required by Serializable / BaseMemory)
// ---------------------------------------------------------------------------

interface DakeraMemorySnapshot {
  messages: Message[];
  url: string;
  apiKey: string | undefined;
  agentId: string;
  sessionId: string | undefined;
  topK: number;
  persist: boolean;
}

// ---------------------------------------------------------------------------
// DakeraMemory
// ---------------------------------------------------------------------------

/**
 * BeeAI memory implementation backed by the Dakera persistent memory API.
 *
 * The local `messages` array serves as the in-context sliding window that the
 * LLM sees.  On every `add()` call the message is also written to Dakera so
 * it survives across sessions.  Before the LLM call the window is prepended
 * with the top-K semantically similar memories retrieved from Dakera — this
 * gives the agent long-term recall without blowing up the context window.
 */
export class DakeraMemory extends BaseMemory<DakeraMemorySnapshot> {
  public messages: Message[] = [];

  private readonly url: string;
  private readonly apiKey: string | undefined;
  private readonly agentId: string;
  private readonly sessionId: string | undefined;
  private readonly topK: number;
  private readonly persist: boolean;

  /** Dakera memory IDs written in this session — used by forgetSession() */
  private readonly storedIds: string[] = [];

  constructor(input: DakeraMemoryInput) {
    super();
    if (!input.url) {
      throw new MemoryError("DakeraMemory: `url` is required.");
    }
    if (!input.agentId) {
      throw new MemoryError("DakeraMemory: `agentId` is required.");
    }
    this.url = input.url.replace(/\/$/, ""); // strip trailing slash
    this.apiKey = input.apiKey;
    this.agentId = input.agentId;
    this.sessionId = input.sessionId;
    this.topK = input.topK ?? 5;
    this.persist = input.persist ?? true;
  }

  // Registration with the BeeAI serializer (mirrors pattern in all built-ins)
  static {
    this.register();
  }

  // ---------------------------------------------------------------------------
  // BaseMemory abstract methods
  // ---------------------------------------------------------------------------

  /**
   * Add a message to the local window and, if `persist` is true, store it in
   * Dakera then inject the top-K semantically relevant past memories at the
   * front of the window.
   */
  async add(message: Message, index?: number): Promise<void> {
    // 1. Insert into the local in-context window at the requested position.
    const insertAt = index !== undefined ? this.clampIndex(index) : this.messages.length;
    this.messages.splice(insertAt, 0, message);

    if (!this.persist || !message.text?.trim()) {
      return;
    }

    // 2. Persist to Dakera.
    try {
      const stored = await this.dakeraStore(message);
      if (stored?.id) {
        this.storedIds.push(stored.id);
      }
    } catch (err) {
      // Non-fatal: remote storage failure should not crash the agent.
      console.warn("[DakeraMemory] Failed to persist message:", err);
    }

    // 3. Retrieve top-K semantically relevant memories and inject them at the
    //    very beginning of the window (before all other messages) so the LLM
    //    always has long-term context available.
    if (this.topK > 0) {
      try {
        await this.injectRecalledMemories(message.text);
      } catch (err) {
        console.warn("[DakeraMemory] Failed to retrieve memories:", err);
      }
    }
  }

  async delete(message: Message): Promise<boolean> {
    const idx = this.messages.indexOf(message);
    if (idx === -1) {
      return false;
    }
    this.messages.splice(idx, 1);
    return true;
  }

  reset(): void {
    this.messages.length = 0;
  }

  // ---------------------------------------------------------------------------
  // Snapshot / serialization (required by Serializable)
  // ---------------------------------------------------------------------------

  createSnapshot(): DakeraMemorySnapshot {
    return {
      messages: [...this.messages],
      url: this.url,
      apiKey: this.apiKey,
      agentId: this.agentId,
      sessionId: this.sessionId,
      topK: this.topK,
      persist: this.persist,
    };
  }

  loadSnapshot(state: DakeraMemorySnapshot): void {
    Object.assign(this, {
      messages: state.messages,
      url: state.url,
      apiKey: state.apiKey,
      agentId: state.agentId,
      sessionId: state.sessionId,
      topK: state.topK,
      persist: state.persist,
    });
  }

  // ---------------------------------------------------------------------------
  // Dakera-specific public helpers
  // ---------------------------------------------------------------------------

  /**
   * Explicitly search Dakera for memories matching `query`.
   * Returns the raw Dakera search hits (with scores) rather than Message objects
   * so callers can decide what to do with them.
   */
  async search(query: string, topK?: number): Promise<DakeraSearchHit[]> {
    return this.dakeraSearch(query, topK ?? this.topK);
  }

  /**
   * Delete all memories written in the current session from Dakera.
   * The local `messages` window is NOT affected.
   */
  async forgetSession(): Promise<void> {
    if (this.storedIds.length === 0) {
      return;
    }
    await this.dakeraForget({ memory_ids: [...this.storedIds] });
    this.storedIds.length = 0;
  }

  // ---------------------------------------------------------------------------
  // Private — Dakera HTTP helpers
  // ---------------------------------------------------------------------------

  private buildHeaders(): Record<string, string> {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };
    if (this.apiKey) {
      headers["Authorization"] = `Bearer ${this.apiKey}`;
    }
    return headers;
  }

  private async dakeraStore(message: Message): Promise<DakeraMemoryRecord | null> {
    const body: DakeraStoreRequest = {
      content: message.text,
      agent_id: this.agentId,
      ...(this.sessionId ? { session_id: this.sessionId } : {}),
      metadata: {
        role: message.role,
        messageId: message.id,
        createdAt: message.meta?.createdAt?.toISOString(),
      },
    };

    const response = await fetch(`${this.url}/v1/memory/store`, {
      method: "POST",
      headers: this.buildHeaders(),
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const text = await response.text().catch(() => "(unreadable body)");
      throw new MemoryError(
        `Dakera store failed: HTTP ${response.status} ${response.statusText} — ${text}`,
      );
    }

    const json = (await response.json()) as DakeraStoreResponse;
    return json?.memory ?? null;
  }

  private async dakeraSearch(query: string, topK: number): Promise<DakeraSearchHit[]> {
    const body: DakeraSearchRequest = {
      agent_id: this.agentId,
      query,
      top_k: topK,
      ...(this.sessionId ? { session_id: this.sessionId } : {}),
    };

    const response = await fetch(`${this.url}/v1/memory/search`, {
      method: "POST",
      headers: this.buildHeaders(),
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const text = await response.text().catch(() => "(unreadable body)");
      throw new MemoryError(
        `Dakera search failed: HTTP ${response.status} ${response.statusText} — ${text}`,
      );
    }

    const json = (await response.json()) as DakeraSearchResponse;
    return json?.memories ?? [];
  }

  private async dakeraForget(opts: Omit<DakeraForgetRequest, "agent_id">): Promise<void> {
    const body: DakeraForgetRequest = {
      agent_id: this.agentId,
      ...opts,
    };

    const response = await fetch(`${this.url}/v1/memory/forget`, {
      method: "POST",
      headers: this.buildHeaders(),
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const text = await response.text().catch(() => "(unreadable body)");
      throw new MemoryError(
        `Dakera forget failed: HTTP ${response.status} ${response.statusText} — ${text}`,
      );
    }
  }

  // ---------------------------------------------------------------------------
  // Private — injection helpers
  // ---------------------------------------------------------------------------

  /**
   * Retrieve top-K memories from Dakera and prepend them to `this.messages`
   * as SystemMessage-role entries marked with a special tag so they can be
   * identified and de-duplicated on subsequent calls.
   *
   * We use Message.of() so the messages are proper framework objects that the
   * LLM runner can serialize.  The role is "system" so the LLM treats them
   * as background context rather than conversation turns.
   */
  private async injectRecalledMemories(query: string): Promise<void> {
    const hits = await this.dakeraSearch(query, this.topK);
    if (hits.length === 0) {
      return;
    }

    // Remove any previously injected recalled-memory messages to avoid
    // accumulating stale injections across multiple add() calls.
    this.purgeInjected();

    // Build recalled memory block as a single system message so it occupies
    // one slot in the context rather than N individual messages.
    const block = hits
      .map((h, i) => `[Memory ${i + 1}] ${h.memory.content}`)
      .join("\n");

    const injected = Message.of({
      role: "system",
      text: `Relevant memories from past sessions:\n${block}`,
      meta: { __dakera_injected: true },
    });

    // Insert at position 0 so it appears before the conversation history.
    this.messages.splice(0, 0, injected);
  }

  /**
   * Remove messages that were injected by a previous `injectRecalledMemories`
   * call (identified by the `__dakera_injected` flag in their meta).
   */
  private purgeInjected(): void {
    for (let i = this.messages.length - 1; i >= 0; i--) {
      if (this.messages[i].meta?.__dakera_injected === true) {
        this.messages.splice(i, 1);
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Private — index utilities
  // ---------------------------------------------------------------------------

  private clampIndex(index: number): number {
    const len = this.messages.length;
    if (index < 0) {
      return Math.max(len + index, 0);
    }
    return Math.min(index, len);
  }
}
