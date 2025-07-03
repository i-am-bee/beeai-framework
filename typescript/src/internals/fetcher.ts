/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { FrameworkError } from "@/errors.js";
import { Serializable } from "@/internals/serializable.js";
import {
  EventSourceMessage,
  EventStreamContentType,
  fetchEventSource,
} from "@ai-zen/node-fetch-event-source";
import { FetchEventSourceInit } from "@ai-zen/node-fetch-event-source/lib/cjs/fetch.js";
import { emitterToGenerator } from "@/internals/helpers/promise.js";
import { doNothing, isArray, isPlainObject, isTruthy } from "remeda";
import { Callback, Emitter } from "@/emitter/emitter.js";
import { shallowCopy } from "@/serializer/utils.js";

export class RestfulClientError extends FrameworkError {}

type URLParamType = string | number | boolean | null | undefined;
export function createURLParams(
  data: Record<string, URLParamType | URLParamType[] | Record<string, any>>,
) {
  const urlTokenParams = new URLSearchParams();
  for (const [key, value] of Object.entries(data)) {
    if (value === undefined) {
      continue;
    }

    if (Array.isArray(value)) {
      value.forEach((v) => {
        if (v !== undefined) {
          urlTokenParams.append(key, String(v));
        }
      });
    } else if (isPlainObject(value)) {
      urlTokenParams.set(key, createURLParams(value).toString());
    } else {
      urlTokenParams.set(key, String(value));
    }
  }
  return urlTokenParams;
}

interface FetchInput {
  url: string;
  options: RequestInit;
}

export interface StreamInput {
  url: string;
  options: FetchEventSourceInit;
}

export interface RestfulClientEvents {
  fetchStart: Callback<{ input: FetchInput }>;
  fetchError: Callback<{ error: Error; input: FetchInput }>;
  fetchSuccess: Callback<{ response: Response; data: any; input: FetchInput }>;
  fetchDone: Callback<{ input: FetchInput }>;

  streamStart: Callback<{ input: StreamInput }>;
  streamOpen: Callback<{ input: StreamInput }>;
  streamSuccess: Callback<{ input: StreamInput }>;
  streamMessage: Callback<{ data: EventSourceMessage; input: StreamInput }>;
  streamError: Callback<{ error: Error; input: StreamInput }>;
  streamDone: Callback<{ input: StreamInput }>;
}

export class RestfulClient<K extends Record<string, string>> extends Serializable {
  public readonly emitter = Emitter.root.child<RestfulClientEvents>({
    namespace: ["internals", "restfulClient"],
    creator: this,
  });

  constructor(
    protected input: {
      baseUrl: string;
      headers?: () => Promise<HeadersInit>;
      paths: K;
    },
  ) {
    super();
  }

  async *stream(
    path: keyof K,
    init: FetchEventSourceInit,
  ): AsyncGenerator<EventSourceMessage, void, void> {
    const emitter = this.emitter.child({
      groupId: "stream",
    });

    const input: StreamInput = {
      url: this.getUrl(path).toString(),
      options: {
        method: "POST",
        ...init,
        headers: await this.getHeaders(init?.headers),
      },
    };
    await emitter.emit("streamStart", { input });

    return yield* emitterToGenerator(async ({ emit }) =>
      fetchEventSource(input.url, {
        ...input.options,
        async onopen(response) {
          const contentType = response.headers.get("content-type") || "";
          if (response.ok && contentType.includes(EventStreamContentType)) {
            await emitter.emit("streamOpen", { input });
            return;
          }
          throw new RestfulClientError("Failed to stream!", [], {
            context: {
              url: response.url,
              err: await response.text(),
              response,
            },
            isRetryable: response.status >= 400 && response.status < 500 && response.status !== 429,
          });
        },
        async onmessage(msg) {
          if (msg?.event === "error") {
            throw new RestfulClientError(`Error during streaming has occurred.`, [], {
              context: msg,
            });
          }
          await emitter.emit("streamMessage", { input, data: msg });
          emit(msg);
        },
        onclose() {},
        onerror(err) {
          throw new RestfulClientError(`Error during streaming has occurred.`, [err]);
        },
      })
        .then(() => emitter.emit("streamSuccess", { input }))
        .catch(async (error) => {
          await emitter.emit("streamError", { input, error }).catch(doNothing());
          throw error;
        })
        .finally(() => emitter.emit("streamDone", { input })),
    );
  }

  async fetch(path: keyof K, init?: RequestInit & { searchParams?: URLSearchParams }) {
    const emitter = this.emitter.child({
      groupId: "fetch",
    });

    const target = this.getUrl(path);
    if (init?.searchParams) {
      for (const [key, value] of init.searchParams) {
        target.searchParams.set(key, value);
      }
    }

    const input: FetchInput = {
      url: target.toString(),
      options: {
        ...init,
        headers: await this.getHeaders(init?.headers),
      },
    };

    await emitter.emit("fetchStart", { input });
    try {
      const response = await fetch(input.url, input.options);

      if (!response.ok) {
        throw new RestfulClientError("Fetch has failed", [], {
          context: {
            url: response.url,
            error: await response.text(),
            response,
          },
          isRetryable: [408, 503].includes(response.status ?? 500),
        });
      }

      const data = await response.json();
      await emitter.emit("fetchSuccess", { response, data, input });
      return data;
    } catch (error) {
      await emitter.emit("fetchError", { error, input: input });
      throw error;
    } finally {
      await emitter.emit("fetchDone", { input: input });
    }
  }

  protected async getHeaders(extra?: HeadersInit): Promise<Record<string, string>> {
    const final = {};
    for (const override of [await this.input.headers?.(), extra].filter(isTruthy)) {
      if (isArray(override) || override instanceof Headers) {
        Object.assign(final, Object.fromEntries(override.entries()));
      } else {
        Object.assign(final, override);
      }
    }
    return final;
  }

  protected getUrl(path: keyof K): URL {
    const url = new URL(this.input.baseUrl);
    const extraPath = this.input.paths[path] ?? path;
    if (url.pathname.endsWith("/")) {
      url.pathname += extraPath.replace(/^\//, "");
    } else {
      url.pathname += extraPath;
    }
    return url;
  }

  createSnapshot() {
    return {
      input: shallowCopy(this.input),
      emitter: this.emitter,
    };
  }

  loadSnapshot(snapshot: ReturnType<typeof this.createSnapshot>): void {
    Object.assign(this, snapshot);
  }
}
