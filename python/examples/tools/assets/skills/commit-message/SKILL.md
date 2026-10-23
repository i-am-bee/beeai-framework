---
name: commit-message
description: >-
  Use when writing or reviewing a git commit message. Defines the type prefix,
  subject line limit and body conventions for this project.
---

# Writing a commit message

- Format the subject as `<type>(<scope>): <subject>`, where `type` is one of
  `feat`, `fix`, `docs`, `refactor`, `test`, `chore`.
- Keep the subject imperative, lowercase, without a trailing period, and at most
  72 characters including the prefix.
- Leave exactly one blank line before the body.
- Use the body to explain *why* the change is needed, wrapped at 72 characters.
  Do not repeat the diff, and do not use bullet lists.
- Reference the issue once, in a final `Refs: #123` line.
