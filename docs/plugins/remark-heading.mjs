import { visit } from "unist-util-visit";

/**
 * Mintlify uses `<Heading level={2} id="anchor">Text</Heading>` for headings
 * that need an explicit anchor id. Astro/Starlight build the table of contents
 * from real Markdown heading nodes, so we convert these JSX elements back into
 * Markdown headings, preserving the explicit id (TOC + deep links keep working).
 */
function getAttr(node, name) {
  const attr = (node.attributes || []).find(
    (a) => a.type === "mdxJsxAttribute" && a.name === name,
  );
  if (!attr) return undefined;
  if (typeof attr.value === "string") return attr.value;
  if (attr.value && typeof attr.value === "object" && "value" in attr.value) {
    return attr.value.value;
  }
  return undefined;
}

export default function remarkHeading() {
  return (tree) => {
    visit(tree, (node, index, parent) => {
      if (!parent || index === null || index === undefined) return;
      if (
        (node.type === "mdxJsxFlowElement" ||
          node.type === "mdxJsxTextElement") &&
        node.name === "Heading"
      ) {
        const rawLevel = getAttr(node, "level");
        const level = Number.parseInt(rawLevel ?? "2", 10) || 2;
        const id = getAttr(node, "id");
        const depth = Math.min(Math.max(level, 1), 6);

        const heading = {
          type: "heading",
          depth,
          children: node.children || [],
        };
        if (id) {
          // hProperties.id sets the rendered heading's id (anchor + TOC).
          heading.data = { hProperties: { id } };
        }
        parent.children[index] = heading;
      }
    });
  };
}
