/**
 * Migrate the archived Mintlify docs (../docs-old) into the Starlight content
 * collection (src/content/docs), preserving the file path -> URL structure.
 *
 * Page bodies are copied VERBATIM — Mintlify components are handled at build
 * time by the shims (src/components/mintlify) and remark plugins (plugins/).
 * Only the frontmatter is normalized for Starlight:
 *   - `sidebarTitle: X`  -> `sidebar:\n  label: X`
 *   - `title` / `description` / `icon` are preserved (icon allowed via schema)
 *   - files without a title get one from their first H1, else the filename
 *
 * Re-runnable: overwrites the content dir from ../docs-old each time.
 *
 * Usage:  node scripts/migrate-content.mjs
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { redirects as REDIRECTS } from "../redirects.mjs";

const here = path.dirname(fileURLToPath(import.meta.url));
const SRC = path.resolve(here, "../../docs-old");
const OUT = path.resolve(here, "../src/content/docs");

// Directories/files in ../docs-old that are not documentation pages.
const SKIP_DIRS = new Set(["node_modules", "snippets", "logo", ".idea", ".yarn", ".mypy_cache"]);
// Repo-internal files that were never public docs pages (not in nav, unlinked).
const SKIP_FILES = new Set(["CONTRIBUTING.md"]);
const PAGE_EXT = new Set([".md", ".mdx"]);

function walk(dir, acc = []) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    if (entry.name.startsWith(".")) continue;
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      if (SKIP_DIRS.has(entry.name)) continue;
      walk(full, acc);
    } else if (PAGE_EXT.has(path.extname(entry.name)) && !SKIP_FILES.has(entry.name)) {
      acc.push(full);
    }
  }
  return acc;
}

function parseFrontmatter(text) {
  const m = text.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n?/);
  if (!m) return { data: {}, body: text };
  const data = {};
  for (const line of m[1].split(/\r?\n/)) {
    const kv = line.match(/^([A-Za-z][\w-]*):\s*(.*)$/);
    if (kv) data[kv[1]] = kv[2].trim().replace(/^["']|["']$/g, "");
  }
  return { data, body: text.slice(m[0].length) };
}

// Mintlify resolved relative `.mdx` links against the file's directory and its
// pages had no trailing slash; Starlight pages do, which breaks those links.
// Rewrite every relative markdown link to a root-absolute URL (extension
// stripped), so links resolve regardless of trailing-slash behaviour.
function absolutizeLinks(body, pageDir) {
  return body.replace(
    /\]\((\.\.?\/[^)\s#]*)(#[^)\s]*)?\)/g,
    (_m, target, hash) => {
      const noExt = target.replace(/\.(mdx?|md)$/i, "");
      let abs = path.posix.normalize(path.posix.join(pageDir, noExt));
      if (REDIRECTS[abs]) abs = REDIRECTS[abs];
      return `](${abs}${hash || ""})`;
    },
  );
}

// Targeted fixes for links that were already broken in the Mintlify source
// (surfaced by starlight-links-validator):
//  - tools.mdx links to #think / #handoff sections that don't exist -> de-link
//  - serve.mdx links to agent-stack#server; the real anchor slug differs
const LINK_FIXUPS = [
  [/\[Think\]\(#think\)/g, "Think"],
  [/\[Handoff\]\(#handoff\)/g, "Handoff"],
  [
    /\/integrations\/agent-stack\/#server/g,
    "/integrations/agent-stack/#exposing-to-the-platform-server",
  ],
];

// The site uses `trailingSlash: 'never'` to match the old Mintlify URLs, so
// strip any trailing slash from absolute internal links (the source authored a
// few like `/modules/tools/` and `/integrations/a2a/#server`). Leaves the bare
// root `/` and external links untouched.
function stripTrailingSlashes(body) {
  return body.replace(
    /\]\((\/[^)\s#]+?)\/(#[^)\s]*)?\)/g,
    (_m, p, hash) => `](${p}${hash || ""})`,
  );
}

function fixLinks(body, pageDir) {
  let out = absolutizeLinks(body, pageDir);
  for (const [re, repl] of LINK_FIXUPS) out = out.replace(re, repl);
  return stripTrailingSlashes(out);
}

function titleFromFilename(file) {
  return path
    .basename(file, path.extname(file))
    .replace(/[-_]/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function buildFrontmatter(data) {
  const lines = ["---"];
  lines.push(`title: ${JSON.stringify(data.title)}`);
  if (data.description) lines.push(`description: ${JSON.stringify(data.description)}`);
  if (data.sidebarTitle) {
    lines.push("sidebar:");
    lines.push(`  label: ${JSON.stringify(data.sidebarTitle)}`);
  }
  if (data.icon) lines.push(`icon: ${JSON.stringify(data.icon)}`);
  lines.push("---", "");
  return lines.join("\n");
}

let migrated = 0;
for (const file of walk(SRC)) {
  const rel = path.relative(SRC, file);
  const text = fs.readFileSync(file, "utf8");
  let { data, body } = parseFrontmatter(text);

  if (!data.title) {
    const h1 = body.match(/^#\s+(.+?)\s*$/m);
    if (h1) {
      data.title = h1[1].trim();
      // Drop the H1 so it isn't duplicated under Starlight's page title.
      body = body.replace(/^#\s+.+?\s*$\r?\n?/m, "");
    } else {
      data.title = titleFromFilename(file);
    }
  }

  const slug = rel.replace(/\.(mdx?|md)$/i, "");
  const dir = path.posix.dirname(slug);
  body = fixLinks(body, dir === "." ? "/" : "/" + dir);

  const dest = path.join(OUT, rel);
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  fs.writeFileSync(dest, buildFrontmatter(data) + body.replace(/^\n+/, ""));
  migrated++;
}

console.log(`Migrated ${migrated} pages from ${SRC} -> ${OUT}`);
