/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { timingSafeEqual } from "node:crypto";

/** Matches the HTTP "Bearer" auth scheme only at the start of the header value. */
const BearerPrefix = /^Bearer\s+/i;

/**
 * Check a caller-supplied API key against the configured one.
 *
 * Returns `true` when no key is configured, meaning authentication is disabled.
 *
 * The comparison runs in constant time with respect to the secret's contents. A plain
 * `!==` short-circuits on the first differing character and can leak the key to an
 * attacker who can measure response latency.
 */
export function isApiKeyValid(
  expected: string | undefined | null,
  received: string | undefined | null,
  { stripBearerPrefix = false }: { stripBearerPrefix?: boolean } = {},
): boolean {
  if (!expected) {
    return true;
  }
  if (!received) {
    return false;
  }

  const token = stripBearerPrefix ? received.replace(BearerPrefix, "") : received;

  const a = Buffer.from(token, "utf8");
  const b = Buffer.from(expected, "utf8");

  // `timingSafeEqual` throws unless both buffers are the same length, so the lengths are
  // compared first. That makes the key's *length* observable, but not its contents, which
  // matches the guarantee Python's `hmac.compare_digest` gives on the other side.
  if (a.length !== b.length) {
    return false;
  }

  return timingSafeEqual(a, b);
}
