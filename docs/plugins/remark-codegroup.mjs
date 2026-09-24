import { visit } from "unist-util-visit";

/**
 * Mintlify groups code blocks with `<CodeGroup>` and reads each block's tab
 * label from its fence info string, e.g.:
 *
 *   <CodeGroup>
 *   ```py Python [expandable]
 *   ...
 *   ```
 *   ```ts TypeScript
 *   ...
 *   ```
 *   </CodeGroup>
 *
 * Starlight has no `<CodeGroup>`, but it ships `<Tabs>` / `<TabItem>`. This
 * plugin rewrites each `<CodeGroup>` into `<Tabs>` with one `<TabItem>` per
 * fenced block, moving the label out of the fence meta and onto the TabItem.
 * Mintlify-only meta flags such as `[expandable]` are stripped.
 */

const LANG_LABELS = new Set([
  "python", "py", "typescript", "ts", "javascript", "js", "jsx", "tsx",
  "node", "nodejs", "java", "go", "golang", "rust", "rs", "ruby", "rb",
  "php", "csharp", "c#", "cpp", "c++", "kotlin", "swift", "scala",
]);

const PRETTY_LANG = {
  py: "Python",
  python: "Python",
  ts: "TypeScript",
  typescript: "TypeScript",
  js: "JavaScript",
  javascript: "JavaScript",
  sh: "Shell",
  bash: "Bash",
  shell: "Shell",
  json: "JSON",
  yaml: "YAML",
  toml: "TOML",
};

function cleanLabel(meta) {
  if (!meta) return "";
  // Drop Mintlify-only bracketed flags like [expandable] / [highlight].
  return meta.replace(/\[[^\]]*\]/g, "").trim();
}

function labelFor(code) {
  const fromMeta = cleanLabel(code.meta);
  if (fromMeta) return fromMeta;
  const lang = (code.lang || "").toLowerCase();
  return PRETTY_LANG[lang] || code.lang || "Code";
}

export default function remarkCodegroup() {
  return (tree) => {
    visit(tree, (node, index, parent) => {
      if (
        node.type !== "mdxJsxFlowElement" ||
        node.name !== "CodeGroup" ||
        !parent ||
        index === null ||
        index === undefined
      ) {
        return;
      }

      const children = node.children || [];
      const codes = children.filter((c) => c.type === "code");
      // Genuine content (prose, lists, ...) an author may have placed alongside
      // the code blocks — preserved before the tabs rather than discarded. Only
      // embedme comments and blank whitespace are dropped.
      const extras = children.filter(
        (c) =>
          c.type !== "code" &&
          c.type !== "mdxFlowExpression" &&
          c.type !== "mdxTextExpression" &&
          !(c.type === "text" && !c.value.trim()),
      );

      // No code inside: drop the wrapper, keep any genuine content children.
      if (codes.length === 0) {
        parent.children.splice(index, 1, ...extras);
        return;
      }

      const labels = codes.map(labelFor);
      // Strip the now-redundant fence info so Expressive Code doesn't choke.
      for (const code of codes) code.meta = null;

      // A single block needs no tabs.
      if (codes.length === 1) {
        parent.children.splice(index, 1, ...extras, codes[0]);
        return;
      }

      const tabItems = codes.map((code, i) => ({
        type: "mdxJsxFlowElement",
        name: "TabItem",
        attributes: [
          { type: "mdxJsxAttribute", name: "label", value: labels[i] },
        ],
        children: [code],
      }));

      const allLanguages = labels.every((l) =>
        LANG_LABELS.has(l.toLowerCase()),
      );
      const attributes = allLanguages
        ? [{ type: "mdxJsxAttribute", name: "syncKey", value: "lang" }]
        : [];

      parent.children.splice(index, 1, ...extras, {
        type: "mdxJsxFlowElement",
        name: "Tabs",
        attributes,
        children: tabItems,
      });
    });
  };
}
