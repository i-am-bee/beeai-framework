// @ts-check
import { defineConfig } from "astro/config";
import { unified } from "@astrojs/markdown-remark";
import starlight from "@astrojs/starlight";
import starlightLinksValidator from "starlight-links-validator";
import mdx from "@astrojs/mdx";

import remarkHeading from "./plugins/remark-heading.mjs";
import remarkCodegroup from "./plugins/remark-codegroup.mjs";
import remarkAutoImport from "./plugins/remark-auto-import.mjs";
import remarkBaseLinks from "./plugins/remark-base-links.mjs";
import { redirects } from "./redirects.mjs";

// Served from the GitHub Pages project URL (https://i-am-bee.github.io/beeai-framework/),
// so everything lives under this base path. When the custom domain
// (framework.beeai.dev) is wired up later, set BASE back to "/" and switch
// `site` to the custom domain — the link/redirect prefixing below no-ops at "/".
const BASE = "/beeai-framework";

// Prefix an internal absolute path with BASE (Astro does not do this for
// redirect targets, just as it doesn't for hand-written content links).
const withBase = (path) => (BASE === "/" ? path : `${BASE}${path}`);

// https://astro.build/config
export default defineConfig({
  site: "https://i-am-bee.github.io",
  base: BASE,

  // Match the previous Mintlify URLs exactly: no trailing slash, and emit
  // `page.html` (served at `/page`) instead of `page/index.html` (`/page/`).
  trailingSlash: "never",
  build: { format: "file" },

  // Mintlify-compatibility build-time transforms. Order matters:
  // headings first, then CodeGroup -> Tabs, then inject the needed imports.
  markdown: {
    processor: unified({
      remarkPlugins: [
        remarkHeading,
        remarkCodegroup,
        remarkAutoImport,
        // Must run last: rewrites root-relative content links/images to include
        // the base path. Tuple form so unified passes the options to the factory.
        [remarkBaseLinks, { base: BASE }],
      ],
    }),
  },

  redirects: {
    // Site root -> first documentation page (matches the old entry point).
    "/": withBase("/introduction/welcome"),
    // Page redirects ported from docs.json (shared with migrate-content.mjs).
    // Targets need the base prefix; Astro doesn't add it to redirect destinations.
    ...Object.fromEntries(
      Object.entries(redirects).map(([from, to]) => [from, withBase(to)]),
    ),
  },

  integrations: [
    starlight({
      title: "BeeAI Framework",
      // Validates internal links & heading anchors at build time
      // (replaces Mintlify's `broken-links` check; also runs in CI).
      // Localhost URLs are allowed — they appear as example endpoints in prose.
      plugins: [starlightLinksValidator({ errorOnLocalLinks: false })],
      logo: {
        light: "./src/assets/beeai-framework-light.svg",
        dark: "./src/assets/beeai-framework-dark.svg",
        replacesTitle: true,
      },
      favicon: "/favicon.svg",
      customCss: ["./src/styles/custom.css"],
      editLink: {
        baseUrl: "https://github.com/i-am-bee/beeai-framework/edit/main/docs/",
      },
      // Ported from docs.json "navbar.links" + "footer.socials".
      social: [
        {
          icon: "github",
          label: "GitHub",
          href: "https://github.com/i-am-bee/beeai-framework",
        },
        {
          icon: "discord",
          label: "Discord",
          href: "https://discord.gg/NradeA6ZNF",
        },
        {
          icon: "blueSky",
          label: "Bluesky",
          href: "https://bsky.app/profile/beeaiagents.bsky.social",
        },
        {
          icon: "youtube",
          label: "YouTube",
          href: "https://www.youtube.com/@BeeAIAgents",
        },
      ],
      // Ported verbatim from docs.json "navigation.groups".
      sidebar: [
        {
          label: "Introduction",
          items: [
            "introduction/welcome",
            "introduction/quickstart",
            "introduction/tour",
          ],
        },
        {
          label: "Core Concepts",
          items: [
            "modules/agents/requirement-agent",
            "modules/middleware",
            "modules/agents",
            "modules/serve",
            "modules/backend",
            "modules/tools",
            "modules/memory",
            "modules/rag",
            "modules/observability",
            "modules/cache",
            "modules/logger",
            "modules/serialization",
            "modules/errors",
            "modules/templates",
            "modules/workflows",
          ],
        },
        {
          label: "Integrations",
          items: [
            "integrations/agent-stack",
            "integrations/mcp",
            "integrations/a2a",
            "integrations/acp-zed",
            "integrations/watsonx-orchestrate",
            "integrations/openai-api",
          ],
        },
        {
          label: "Guides",
          items: ["guides/mcp-slackbot"],
        },
        {
          label: "Community",
          items: ["community/contribute"],
        },
      ],
    }),
    mdx(),
  ],

  vite: {
    resolve: {
      alias: {
        "@components": new URL("./src/components", import.meta.url).pathname,
      },
    },
  },
});
