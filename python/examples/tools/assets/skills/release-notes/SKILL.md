---
name: release-notes
description: >-
  Use when writing release notes for a version. Defines the section order,
  bullet style and required footer for this project's releases.
---

# Writing release notes

1. Group every change under one of these headings, in this order:
   `### Added`, `### Changed`, `### Fixed`, `### Removed`. Drop empty sections.
2. Write one bullet per user-facing change, imperative mood, no trailing period.
3. Do not include commit hashes or issue numbers.
4. Ignore purely internal changes (refactors, test-only or build-only work) — they
   are not release notes.
5. End with a `## Upgrade notes` section that says `No action required.` when there
   is nothing to do, otherwise a numbered list of migration steps.
6. Keep the whole document under 20 lines.
