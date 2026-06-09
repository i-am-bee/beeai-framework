import { visit } from "unist-util-visit";

/**
 * Mintlify makes its components globally available without imports. To keep the
 * migrated `.mdx` files free of boilerplate import lines, this plugin scans each
 * file for the component names it actually uses and injects the matching import
 * statement(s) at the top of the document.
 *
 * Must run AFTER remark-codegroup (which introduces <Tabs>/<TabItem>).
 */

const SOURCES = [
  {
    source: "@astrojs/starlight/components",
    names: new Set(["Tabs", "TabItem"]),
  },
  {
    source: "@components/mintlify",
    names: new Set([
      "Note", "Tip", "Info", "Warning",
      "Card", "CardGroup",
      "Steps", "Step",
      "Accordion", "AccordionGroup",
      "Rule",
    ]),
  },
];

function buildImportNode(names, source) {
  const value = `import { ${names.join(", ")} } from ${JSON.stringify(source)};`;
  return {
    type: "mdxjsEsm",
    value,
    data: {
      estree: {
        type: "Program",
        sourceType: "module",
        body: [
          {
            type: "ImportDeclaration",
            specifiers: names.map((name) => ({
              type: "ImportSpecifier",
              imported: { type: "Identifier", name },
              local: { type: "Identifier", name },
            })),
            source: { type: "Literal", value: source, raw: JSON.stringify(source) },
            attributes: [],
          },
        ],
      },
    },
  };
}

export default function remarkAutoImport() {
  return (tree) => {
    const used = new Set();
    visit(tree, (node) => {
      if (
        (node.type === "mdxJsxFlowElement" ||
          node.type === "mdxJsxTextElement") &&
        node.name
      ) {
        used.add(node.name);
      }
    });
    if (used.size === 0) return;

    // Names already imported by the author, to avoid redeclaration.
    const alreadyImported = new Set();
    for (const node of tree.children) {
      if (node.type === "mdxjsEsm" && node.data?.estree?.body) {
        for (const stmt of node.data.estree.body) {
          if (stmt.type === "ImportDeclaration") {
            for (const spec of stmt.specifiers) {
              if (spec.local?.name) alreadyImported.add(spec.local.name);
            }
          }
        }
      }
    }

    const importNodes = [];
    for (const { source, names } of SOURCES) {
      const needed = [...names].filter(
        (n) => used.has(n) && !alreadyImported.has(n),
      );
      if (needed.length > 0) importNodes.push(buildImportNode(needed, source));
    }

    if (importNodes.length > 0) tree.children.unshift(...importNodes);
  };
}
