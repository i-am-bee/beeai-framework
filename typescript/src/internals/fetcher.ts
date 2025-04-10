/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
import { doNothing, isPlainObject } from "remeda";
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
      headers: () => Promise<Headers>;
      paths: K;
    },
  ) {
    super();
  }

  async *stream(
    path: keyof K,
    init: FetchEventSourceInit,
  ): AsyncGenerator<EventSourceMessage, void, void> {
    const { paths, baseUrl, headers } = this.input;
    const emitter = this.emitter.child({
      groupId: "stream",
    });

    const input: StreamInput = {
      url: new URL(paths[path] ?? path, baseUrl).toString(),
      options: {
        ...init,
        method: "POST",
        headers: await headers().then((raw) =>
          Object.assign(Object.fromEntries(raw.entries()), init?.headers),
        ),
      },
    };
    await emitter.emit("streamStart", { input });

    return yield* emitterToGenerator(async ({ emit }) =>
      fetchEventSource(input.url, {
        ...input.options,
        async onopen(response) {
          if (response.ok && response.headers.get("content-type") === EventStreamContentType) {
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

    const { paths, baseUrl, headers: getHeaders } = this.input;

    const target = new URL(paths[path] ?? path, baseUrl);
    if (init?.searchParams) {
      for (const [key, value] of init.searchParams) {
        target.searchParams.set(key, value);
      }
    }

    const input: FetchInput = {
      url: target.toString(),
      options: {
        ...init,
        headers: await getHeaders().then((raw) =>
          Object.assign(Object.fromEntries(raw.entries()), init?.headers),
        ),
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
