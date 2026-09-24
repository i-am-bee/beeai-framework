/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * DakeraMemory — persistent, decay-weighted vector memory for BeeAI agents
 *
 * Integrates with the self-hosted Dakera REST API (https://dakera.ai).
 *
 * Run Dakera locally with the public `dakera-ai/dakera-deploy` docker-compose
 * (the server needs the object store the compose provisions — a bare
 * `docker run` of the image is not sufficient):
 *   git clone https://github.com/dakera-ai/dakera-deploy
 *   cd dakera-deploy && docker compose up -d   # server listens on :3000
 *
 * Usage:
 *   const memory = new DakeraMemory({ url: "http://localhost:3000", agentId: "my-agent" });
 *   const agent = new ReActAgent({ llm, tools, memory });
 *
 * On every add(), the message is persisted to Dakera and the in-session window is enriched
 * with semantically relevant memories recalled from past sessions.
 */

import { BaseMemory, MemoryError } from "@/memory/base.js";
import { Message } from "@/backend/message.js";
import { Logger } from "@/logger/logger.js";

const logger = Logger.root.child({ name: "DakeraMemory" });

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

interface DakeraRecallRequest {
  agent_id: string;
  query: string;
  top_k?: number;
  session_id?: string;
}

interface DakeraRecallHit {
  memory: DakeraMemoryRecord;
  score: number;
}

interface DakeraRecallResponse {
  memories: DakeraRecallHit[];
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
  /** Base URL of the Dakera server, e.g. "http://localhost:3000" */
  url: string;
  /** Optional API key (`dk-...`) — sent as `Authorization: Bearer <apiKey>` */
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
//
// NOTE: `apiKey` is intentionally excluded from the snapshot — serializing a
// secret into a persisted/logged checkpoint would leak it in cleartext. When a
// snapshot is restored the API key must be re-supplied via the constructor if
// write access to Dakera is needed.
// ---------------------------------------------------------------------------

interface DakeraMemorySnapshot {
  messages: Message[];
  url: string;
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
 * with the top-K semantically similar memories recalled from Dakera — this
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

    if (!message.text?.trim()) {
      return;
    }

    // 2. Persist to Dakera (only when persist=true; skip for read-only mode).
    if (this.persist) {
      try {
        const stored = await this.dakeraStore(message);
        if (stored?.id) {
          this.storedIds.push(stored.id);
        }
      } catch (err) {
        // Non-fatal: remote storage failure should not crash the agent.
        logger.warn({ err }, "Failed to persist message to Dakera");
      }
    }

    // 3. Recall the top-K semantically relevant memories and inject them at the
    //    very beginning of the window (before all other messages) so the LLM
    //    always has long-term context available.
    //    This runs regardless of `persist` — read-only mode still benefits from
    //    recall injection; it just doesn't write new memories to Dakera.
    if (this.topK > 0) {
      try {
        await this.injectRecalledMemories(message.text);
      } catch (err) {
        logger.warn({ err }, "Failed to recall memories from Dakera");
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
    // `apiKey` is deliberately omitted — never serialize a secret.
    return {
      messages: [...this.messages],
      url: this.url,
      agentId: this.agentId,
      sessionId: this.sessionId,
      topK: this.topK,
      persist: this.persist,
    };
  }

  loadSnapshot(state: DakeraMemorySnapshot): void {
    // `apiKey` is not part of the snapshot (see createSnapshot). Deserialization
    // bypasses the constructor (Object.create), so re-establish the non-persisted
    // fields at their defaults: the API key is intentionally dropped (must be
    // re-supplied for write access) and the per-session id list starts empty.
    Object.assign(this, {
      messages: state.messages,
      url: state.url,
      apiKey: undefined,
      agentId: state.agentId,
      sessionId: state.sessionId,
      topK: state.topK,
      persist: state.persist,
      storedIds: [],
    });
  }

  // ---------------------------------------------------------------------------
  // Dakera-specific public helpers
  // ---------------------------------------------------------------------------

  /**
   * Explicitly recall memories from Dakera matching `query`.
   * Returns the raw Dakera recall hits (with scores) rather than Message objects
   * so callers can decide what to do with them.
   */
  async recall(query: string, topK?: number): Promise<DakeraRecallHit[]> {
    return this.dakeraRecall(query, topK ?? this.topK);
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
        createdAt:
          message.meta?.createdAt instanceof Date
            ? message.meta.createdAt.toISOString()
            : typeof message.meta?.createdAt === "string"
              ? message.meta.createdAt
              : undefined,
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

  private async dakeraRecall(query: string, topK: number): Promise<DakeraRecallHit[]> {
    const body: DakeraRecallRequest = {
      agent_id: this.agentId,
      query,
      top_k: topK,
      ...(this.sessionId ? { session_id: this.sessionId } : {}),
    };

    const response = await fetch(`${this.url}/v1/memory/recall`, {
      method: "POST",
      headers: this.buildHeaders(),
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const text = await response.text().catch(() => "(unreadable body)");
      throw new MemoryError(
        `Dakera recall failed: HTTP ${response.status} ${response.statusText} — ${text}`,
      );
    }

    const json = (await response.json()) as DakeraRecallResponse;
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
   * Recall top-K memories from Dakera and prepend them to `this.messages`
   * as a SystemMessage-role entry marked with a special tag so it can be
   * identified and de-duplicated on subsequent calls.
   *
   * We use Message.of() so the message is a proper framework object that the
   * LLM runner can serialize.  The role is "system" so the LLM treats it
   * as background context rather than a conversation turn.
   */
  private async injectRecalledMemories(query: string): Promise<void> {
    // Remove any previously injected recalled-memory messages first to avoid
    // accumulating stale injections across multiple add() calls.  Must happen
    // before the recall so that even a no-hit response clears old context.
    this.purgeInjected();

    const hits = await this.dakeraRecall(query, this.topK);
    if (hits.length === 0) {
      return;
    }

    // Build recalled memory block as a single system message so it occupies
    // one slot in the context rather than N individual messages.
    const block = hits.map((h, i) => `[Memory ${i + 1}] ${h.memory.content}`).join("\n");

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
