# Docs migration: Mintlify → Astro Starlight

This directory (`docs/`) is the documentation site, which replaced the Mintlify
site now archived in `../docs-old`. It is built with
**[Astro Starlight](https://starlight.astro.build)**
(MIT, free, static output) and deploys to **GitHub Pages** at the existing
custom domain `framework.beeai.dev`.

> **Status: all pages migrated.** The framework, navigation, redirects, branding,
> search, and the Mintlify-compatibility layer are in place. **All 31 documentation
> pages are ported** (bodies verbatim, frontmatter normalized) and the site builds
> with zero errors. URL structure is preserved 1:1 with the old Mintlify paths, all
> redirects emit, and all internal links resolve. Remaining: embedme wiring, a link
> checker, the GitHub Pages deploy workflow, and cutover (see checklist below).
>
> Run `node scripts/migrate-content.mjs` to re-sync content from `../docs-old`
> until the archive is removed. `CONTRIBUTING.md` (repo-internal dev setup, not in
> nav and unlinked) is intentionally excluded.

## Why Starlight

- Closest UI/UX to the previous Mintlify "maple" theme (clean, minimal, sidebar
  groups, right-hand TOC, dark/light, built-in search via Pagefind).
- MDX-native — keeps the existing content format.
- Near-zero client JS, static HTML → trivial to host on GitHub Pages.

## Hosting model

- Served from the custom domain `framework.beeai.dev` (see `public/CNAME`), so
  the site base path is `/`. **All existing absolute links (`/modules/...`) keep
  working unchanged.**
- `trailingSlash: 'never'` + `build.format: 'file'` emit `page.html` (served at
  `/page`, no trailing slash) so URLs match the previous Mintlify paths **exactly**.
- `public/.nojekyll` ensures GitHub Pages serves Astro's `_astro/` asset dir.

## Node version

Astro 6 requires Node ≥ 22.12. The repo root pins Node 20.15.1 via mise; this
directory overrides it (`docs/mise.toml` + `.nvmrc` → Node 22). Use
`mise install` (or `nvm use`) inside `docs/` before `npm install`.

```bash
cd docs
npm install
npm run dev      # local dev server on :3333
npm run build    # static build into dist/
npm run preview  # serve the built site
```

## The low-touch compatibility layer

Rather than rewrite every Mintlify component by hand, the migration keeps the
`.mdx` bodies almost **verbatim** and handles the differences at build time:

### Shim components (`src/components/mintlify/`)
Drop-in replacements that preserve the Mintlify component API:

| Mintlify | Rendered as |
| --- | --- |
| `<Note>` `<Info>` | Starlight `<Aside type="note">` |
| `<Tip>` | Starlight `<Aside type="tip">` |
| `<Warning>` | Starlight `<Aside type="caution">` |
| `<Card>` / `<CardGroup>` | Starlight `<Card>` / `<CardGrid>` (clickable when `href` is set) |
| `<Steps>` / `<Step title>` | Numbered vertical stepper (CSS in `src/styles/custom.css`) |
| `<Accordion>` / `<AccordionGroup>` | Native `<details>` disclosure |
| `<Rule />` | `<hr>` |

### Remark plugins (`plugins/`)
- **`remark-codegroup.mjs`** — rewrites `<CodeGroup>` + fenced blocks into
  Starlight `<Tabs>` / `<TabItem>`, taking each tab label from the fence info
  string (` ```py Python ` → label "Python"). Mintlify-only flags such as
  `[expandable]` are stripped. Language tabs (Python/TypeScript) get
  `syncKey="lang"` so they switch together; non-language tab sets (e.g.
  macOS/Windows) stay independent — matching Mintlify behaviour.
- **`remark-heading.mjs`** — converts `<Heading level={2} id="x">` back into a
  real Markdown heading, preserving the explicit anchor id so deep links and the
  table of contents keep working.
- **`remark-auto-import.mjs`** — injects the needed component imports per page,
  so the migrated `.mdx` files don't need manual `import` lines.

Net effect: a migrated page is the **original Mintlify MDX with only its
frontmatter adjusted**.

### Frontmatter mapping
- `title`, `description` → unchanged.
- `sidebarTitle: X` → `sidebar:\n  label: X`.
- `icon: ...` (FontAwesome) → kept (schema allows it) but currently unused by the
  sidebar. Can be wired to sidebar icons later if desired.

## Known deltas vs Mintlify (to review)

- **`[expandable]` code blocks** no longer auto-collapse; long code renders in a
  scrollable frame. Can be reproduced with Expressive Code `collapse={…}` if
  wanted.
- **`<CardGroup cols={3}>`** uses Starlight's responsive grid; the 3-up layout is
  applied via CSS on wide viewports (`--mf-cols`).
- **Brand palette** in `src/styles/custom.css` is a warm-grey approximation of
  "maple" — fine-tune to taste.
- **Card icons**: `<Card icon>` is mapped to Starlight's icon set in
  `Card.astro` for the names that have an equivalent (brand icons, `book-open`,
  `check`). FontAwesome names with no Starlight icon (e.g. `python`, `js`,
  `server`) render without an icon rather than failing the build. Add custom
  icons if those cards need them back.
- **Sidebar page icons** (the per-page FontAwesome icons) are not shown by
  default in Starlight.
- A `markdown.*` deprecation warning at build time comes from Starlight's own
  internals (Astro 6.4 API change), not this config; it clears when Starlight
  updates. Removed only in Astro 8.

## Full-migration checklist

1. ~~**Port all pages**~~ — done via `scripts/migrate-content.mjs` (31 pages,
   bodies verbatim, frontmatter normalized).
2. ~~**Verify component coverage**~~ — done; all pages build, 0 leaked tags,
   accordions/tabs/callouts/steppers render. (No `<Frame>`/image usage found.)
3. ~~**embedme**~~ — done. `npm run embedme:verify` / `npm run embedme` operate on
   `src/content/docs/**/*.mdx` with `--source-root=..` (repo root), so snippets keep
   syncing from `python/examples/**` and `typescript/examples/**`. The 3 blocks that
   were stale in the old Mintlify source (all in `modules/agents/requirement-agent.mdx`)
   have been re-synced; `embedme:verify` now passes (145 blocks in sync).
4. ~~**Broken-link check**~~ — done: `starlight-links-validator` runs during the
   build (and in CI), replacing Mintlify's `broken-links`.
5. ~~**GitHub Pages deploy workflow**~~ — done: `.github/workflows/docs-pages.yml`
   builds `docs/` with Node 22 (from `.nvmrc`), gates on `embedme:verify`, and
   publishes via `actions/deploy-pages`. The full CI sequence (`npm ci` →
   `embedme:verify` → `astro build`) was validated locally on Node 22.
6. **Cutover** — in progress: the new site now lives in `docs/`; the old Mintlify
   site is archived in `../docs-old`, and its CI (`.github/workflows/docs.yml`) +
   mise `docs:*` tasks have been retired. Remaining: delete `../docs-old` once the
   deployed site is confirmed, and update any README links.

## One-time GitHub setup (repo admin — cannot be scripted)

The deploy workflow is ready, but Pages must be enabled once in the repo:

1. **Settings → Pages → Build and deployment → Source: "GitHub Actions".**
2. Push to `main` (or run the workflow manually via *Actions → Deploy Docs → Run
   workflow*). The first successful run publishes the site.
3. **Custom domain:** the `framework.beeai.dev` `CNAME` ships in the build output.
   Keep/enter it under Settings → Pages → Custom domain. **DNS cutover** (point the
   `framework.beeai.dev` record away from Mintlify to GitHub Pages —
   `<org>.github.io`) is the final switch; do it once the deployed site looks right.
   Until then, you can preview from the Actions run's deployment URL.
