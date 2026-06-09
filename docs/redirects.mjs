// Page redirects ported from the old Mintlify docs.json.
// Shared by astro.config.mjs (emits redirect pages) and
// scripts/migrate-content.mjs (rewrites links to moved pages to their target).
export const redirects = {
  "/modules/events": "/modules/middleware",
  "/modules/emitter": "/modules/middleware",
  "/experimental/requirement-agent": "/modules/agents/requirement-agent",
  "/integrations/beeai-platform": "/integrations/agent-stack",
};
