// Barrel for the Mintlify-compatibility shim components.
// The remark-auto-import plugin injects a single import from this module
// for whichever of these names a page actually uses.
export { default as Note } from "./Note.astro";
// Mintlify <Info> renders the same as a note aside.
export { default as Info } from "./Note.astro";
export { default as Tip } from "./Tip.astro";
export { default as Warning } from "./Warning.astro";
export { default as Card } from "./Card.astro";
export { default as CardGroup } from "./CardGroup.astro";
export { default as Steps } from "./Steps.astro";
export { default as Step } from "./Step.astro";
export { default as Accordion } from "./Accordion.astro";
export { default as AccordionGroup } from "./AccordionGroup.astro";
export { default as Rule } from "./Rule.astro";
