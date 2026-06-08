import { defineCollection } from "astro:content";
import { docsLoader } from "@astrojs/starlight/loaders";
import { docsSchema } from "@astrojs/starlight/schema";
import { z } from "astro:content";

export const collections = {
  docs: defineCollection({
    loader: docsLoader(),
    // Extend the Starlight schema so that Mintlify frontmatter keys that we
    // intentionally keep around (e.g. `icon`) do not fail validation.
    schema: docsSchema({
      extend: z.object({
        icon: z.string().optional(),
      }),
    }),
  }),
};
