// @ts-check
import { defineConfig } from "astro/config";
import { unified } from "@astrojs/markdown-remark";
import starlight from "@astrojs/starlight";
import starlightLinksValidator from "starlight-links-validator";
import mdx from "@astrojs/mdx";

import remarkHeading from "./plugins/remark-heading.mjs";
import remarkCodegroup from "./plugins/remark-codegroup.mjs";
import remarkAutoImport from "./plugins/remark-auto-import.mjs";
import { redirects } from "./redirects.mjs";

// https://astro.build/config
export default defineConfig({
  // Served from the custom domain (GitHub Pages CNAME), so the base path is "/".
  site: "https://framework.beeai.dev",

  // Mintlify-compatibility build-time transforms. Order matters:
  // headings first, then CodeGroup -> Tabs, then inject the needed imports.
  markdown: {
    processor: unified({
      remarkPlugins: [remarkHeading, remarkCodegroup, remarkAutoImport],
    }),
  },

  redirects: {
    // Site root -> first documentation page (matches the old entry point).
    "/": "/introduction/welcome",
    // Page redirects ported from docs.json (shared with migrate-content.mjs).
    ...redirects,
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
