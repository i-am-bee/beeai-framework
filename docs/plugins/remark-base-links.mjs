import { visit } from "unist-util-visit";

/**
 * Astro applies the configured `base` to links it generates itself (sidebar,
 * nav, asset URLs), but NOT to root-relative links written by hand inside
 * Markdown/MDX content. The migrated content uses Mintlify-style absolute
 * links like `[Backend](/modules/backend)`, which would 404 when the site is
 * served from a sub-path (e.g. https://i-am-bee.github.io/beeai-framework/).
 *
 * This plugin rewrites those root-relative URLs to include the base, covering
 * Markdown links/images, reference definitions, and `href`/`src` on MDX JSX
 * elements (e.g. <Card href="/modules/...">). External URLs, protocol-relative
 * URLs, bare anchors (`#section`), and links that already carry the base are
 * left untouched.
 *
 * When `base` is "/" or empty (i.e. served from a domain root) this is a no-op,
 * so the plugin can stay wired up if the custom domain is restored later.
 */
function makePrefixer(base) {
  // Normalize to no trailing slash so we can prepend cleanly: "/beeai-framework".
  const normalized = base.replace(/\/+$/, "");
  return (url) => {
    if (typeof url !== "string" || url.length === 0) return url;
    // Only root-relative paths; skip "//host", "#anchor", "mailto:", "https://", "./x".
    if (url[0] !== "/" || url[1] === "/") return url;
    // Already prefixed (exact base or base-rooted path) — leave as-is.
    if (url === normalized || url.startsWith(`${normalized}/`)) return url;
    return `${normalized}${url}`;
  };
}

function getStringAttr(node, name) {
  return (node.attributes || []).find(
    (a) =>
      a.type === "mdxJsxAttribute" &&
      a.name === name &&
      typeof a.value === "string",
  );
}

export default function remarkBaseLinks({ base = "/" } = {}) {
  const normalized = base.replace(/\/+$/, "");
  // Nothing to do when served from the domain root.
  if (!normalized) return () => {};

  const prefix = makePrefixer(normalized);

  return (tree) => {
    visit(tree, (node) => {
      // Markdown links/images and reference-style definitions.
      if (
        node.type === "link" ||
        node.type === "image" ||
        node.type === "definition"
      ) {
        node.url = prefix(node.url);
        return;
      }
      // MDX JSX elements: <Card href="/...">, <img src="/..."> etc.
      if (
        node.type === "mdxJsxFlowElement" ||
        node.type === "mdxJsxTextElement"
      ) {
        for (const name of ["href", "src"]) {
          const attr = getStringAttr(node, name);
          if (attr) attr.value = prefix(attr.value);
        }
      }
    });
  };
}
