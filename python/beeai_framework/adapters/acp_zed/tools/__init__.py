# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0
"""The ACP Zed adapter does not ship its own tool subclasses.

Instead, serve adapters install ACP-routed implementations of the generic
`ShellBackend` / `FileBackend` (see `adapters/acp_zed/serve/backends.py`) for the
duration of a turn, so the generic tools from `beeai_framework.tools.code` and
`beeai_framework.tools.filesystem` adapt automatically.
"""
