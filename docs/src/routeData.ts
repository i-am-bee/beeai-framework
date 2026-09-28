/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { defineRouteMiddleware } from "@astrojs/starlight/route-data";

// With `build.format: "file"`, Starlight builds the canonical URL from the emitted file
// name, so every page advertises `/page.html`. The site is served at the extensionless
// `/page` (the form Mintlify used and search engines have indexed, and the form the
// sitemap lists), so rewrite the two head tags that carry the page URL to match it.
const stripHtml = (url: string) => url.replace(/\/index\.html$/, "/").replace(/\.html$/, "");

export const onRequest = defineRouteMiddleware((context) => {
  for (const entry of context.locals.starlightRoute.head) {
    const attrs = entry.attrs;
    if (!attrs) continue;
    if (entry.tag === "link" && attrs.rel === "canonical" && typeof attrs.href === "string") {
      attrs.href = stripHtml(attrs.href);
    } else if (entry.tag === "meta" && attrs.property === "og:url" && typeof attrs.content === "string") {
      attrs.content = stripHtml(attrs.content);
    }
  }
});
