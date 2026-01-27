# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

"""
Agent Scratchpad Tool - Allows agents to track their reasoning and actions.

This tool provides a working memory (scratchpad) where agents can:
- Record actions they've taken
- Store observations/results from tools
- Review their previous reasoning
- Avoid repeating actions
"""

import logging
from typing import ClassVar

from pydantic import BaseModel, Field

from beeai_framework.context import RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.tools import StringToolOutput, Tool, ToolRunOptions

logger = logging.getLogger(__name__)


class ScratchpadInput(BaseModel):
    """Input schema for scratchpad operations."""

    operation: str = Field(
        description=(
            "Operation to perform: 'read' to view scratchpad, "
            "'write' to add entry, 'append' to add to last entry, "
            "'clear' to reset"
        ),
        enum=["read", "write", "append", "clear"],
    )
    content: str | None = Field(
        default=None,
        description=(
            "Content to write/append (required for 'write' and 'append' " "operations)"
        ),
    )


class ScratchpadTool(Tool):
    """Tool for managing agent scratchpad (working memory)."""

    _scratchpads: ClassVar[dict[str, list]] = {}

    def __init__(self, session_id: str | None = None) -> None:
        """Initialize scratchpad tool.

        Args:
            session_id: Optional session identifier (deprecated, not used).
                        Session ID is now extracted from RunContext.
        """
        super().__init__()
        self.middlewares = []

    @staticmethod
    def _ensure_session(session_id: str) -> None:
        """Ensure a session exists in scratchpads."""
        if session_id not in ScratchpadTool._scratchpads:
            ScratchpadTool._scratchpads[session_id] = []

    def _get_session_id(self, context: RunContext | None = None) -> str:
        """Extract session ID from context.

        Always returns "default" to maintain a single persistent scratchpad
        across all requests, ensuring information is retained between interactions.

        Args:
            context: Run context (not used, maintained for compatibility).

        Returns:
            Session ID string (always "default").
        """
        # Use a single persistent session for all operations
        # This ensures the scratchpad persists across HTTP requests
        return "default"

    @property
    def name(self) -> str:
        """Tool name."""
        return "scratchpad"

    @property
    def description(self) -> str:
        """Tool description."""
        return (
            "Manage your working memory (scratchpad). Use this to track "
            "what you've done, what results you got, and avoid repeating "
            "actions. Operations: 'read' - see your scratchpad, 'write' - "
            "add an entry, 'clear' - reset scratchpad, 'append' - add to "
            "existing entry."
        )

    @property
    def input_schema(self) -> type[BaseModel]:
        """Input schema for the tool."""
        return ScratchpadInput

    def _create_emitter(self) -> Emitter:
        """Create emitter for the tool."""
        return Emitter()

    def _get_entries(self, session_id: str) -> list:
        """Get scratchpad entries for a session.

        Args:
            session_id: Session identifier.

        Returns:
            List of scratchpad entries.
        """
        self._ensure_session(session_id)
        return self._scratchpads[session_id]

    def _read_scratchpad(self, session_id: str) -> str:
        """Read the current scratchpad content.

        Args:
            session_id: Session identifier.

        Returns:
            Formatted scratchpad content string.
        """
        entries = self._get_entries(session_id)
        if not entries:
            result = "Scratchpad is empty. No actions recorded yet."
            logger.info(f"ScratchpadTool[{session_id}]: READ - Empty")
            return result

        result = "=== AGENT SCRATCHPAD ===\n\n"
        result += "\n\n".join(f"[{i}] {entry}" for i, entry in enumerate(entries, 1))

        logger.info(f"ScratchpadTool[{session_id}]: READ - {len(entries)} entries")
        return result

    @staticmethod
    def _parse_key_value_pairs(content: str) -> dict:
        """Parse key-value pairs from scratchpad content.

        Handles formats like:
        - "key: value"
        - "key1: value1, key2: value2"
        - "key: value, key2: value2, key3: value3"

        Args:
            content: Content string to parse.

        Returns:
            Dictionary of key-value pairs.
        """
        pairs = {}
        # Split by comma, but be careful with commas inside values
        parts = [p.strip() for p in content.split(",")]
        for part in parts:
            if ":" in part:
                key, value = part.split(":", 1)
                key = key.strip()
                value = value.strip()
                if key and value:
                    pairs[key] = value
        return pairs

    @staticmethod
    def _merge_entries(entries: list, new_pairs: dict) -> list:
        """Merge new key-value pairs into existing entries.

        Args:
            entries: List of existing scratchpad entries.
            new_pairs: Dictionary of new key-value pairs to merge.

        Returns:
            Updated list of entries (consolidated).
        """
        # Parse all existing entries into a single dict
        consolidated = {}
        for entry in entries:
            pairs = ScratchpadTool._parse_key_value_pairs(entry)
            consolidated.update(pairs)

        # Merge new pairs (new values override old ones)
        consolidated.update(new_pairs)

        # Convert back to entry format
        if consolidated:
            # Create a single consolidated entry
            entry_str = ", ".join(f"{k}: {v}" for k, v in consolidated.items())
            return [entry_str]
        return []

    def _write_scratchpad(self, entry: str, session_id: str) -> str:
        """Add or update entry in the scratchpad.

        Merges key-value pairs with existing entries to avoid duplicates.
        If entry contains key-value pairs (format: "key: value"), it will
        update existing entries with the same keys.

        Args:
            entry: Content to add/update.
            session_id: Session identifier.

        Returns:
            Success message.
        """
        entries = self._get_entries(session_id)
        new_pairs = self._parse_key_value_pairs(entry)

        if new_pairs:
            # Merge with existing entries
            entries[:] = self._merge_entries(entries, new_pairs)
            result = f"Updated scratchpad: {', '.join(f'{k}: {v}' for k, v in new_pairs.items())}"
        else:
            # If no key-value pairs found, append as-is (for non-structured entries)
            entries.append(entry)
            result = f"Added to scratchpad: {entry}"

        logger.info(
            f"ScratchpadTool[{session_id}]: WRITE - " f"{len(entries)} total entries"
        )
        return result

    def _append_scratchpad(self, text: str, session_id: str) -> str:
        """Append to the last entry in scratchpad.

        Args:
            text: Text to append.
            session_id: Session identifier.

        Returns:
            Success or error message.
        """
        entries = self._get_entries(session_id)
        if not entries:
            result = "No entry to append to. Use 'write' first."
            logger.info(f"ScratchpadTool[{session_id}]: APPEND - No entries")
            return result

        entries[-1] += f" {text}"
        result = f"Appended to last entry: {text}"
        logger.info(f"ScratchpadTool[{session_id}]: APPEND - Updated")
        return result

    def _clear_scratchpad(self, session_id: str) -> str:
        """Clear the scratchpad.

        Args:
            session_id: Session identifier.

        Returns:
            Success message.
        """
        entries_count = len(self._get_entries(session_id))
        self._scratchpads[session_id] = []
        result = "Scratchpad cleared."
        logger.info(
            f"ScratchpadTool[{session_id}]: CLEAR - " f"{entries_count} entries"
        )
        return result

    async def _run(
        self,
        input: ScratchpadInput,
        options: ToolRunOptions | None = None,
        context: RunContext | None = None,
    ) -> StringToolOutput:
        """Execute scratchpad operation.

        Args:
            input: ScratchpadInput model instance.
            options: Optional tool run options.
            context: Optional run context.

        Returns:
            StringToolOutput with the result of the operation.
        """
        # Get session ID (always "default" for persistent storage)
        session_id = self._get_session_id(context)
        operation = input.operation.lower().strip()
        content = input.content

        logger.info(
            f"ScratchpadTool[{session_id}]: operation='{operation}', "
            f"content='{content}'"
        )

        if not operation:
            error_msg = (
                "Error: 'operation' parameter is required. "
                "Use 'read', 'write', 'append', or 'clear'."
            )
            return StringToolOutput(result=error_msg)

        # Operation handlers
        handlers = {
            "read": lambda: self._read_scratchpad(session_id),
            "write": lambda: (
                self._write_scratchpad(content, session_id)
                if content
                else "Error: 'write' operation requires 'content' parameter."
            ),
            "append": lambda: (
                self._append_scratchpad(content, session_id)
                if content
                else "Error: 'append' operation requires 'content' parameter."
            ),
            "clear": lambda: self._clear_scratchpad(session_id),
        }

        handler = handlers.get(operation)
        if handler:
            result = handler()
            return StringToolOutput(result=result)

        error_msg = (
            f"Unknown operation: {operation}. "
            "Use 'read', 'write', 'append', or 'clear'."
        )
        return StringToolOutput(result=error_msg)

    @classmethod
    def get_scratchpad_for_session(cls, session_id: str) -> list:
        """Get scratchpad entries for a specific session.

        Args:
            session_id: Session identifier.

        Returns:
            List of scratchpad entries.
        """
        return cls._scratchpads.get(session_id, [])

    @classmethod
    def clear_session(cls, session_id: str) -> None:
        """Clear scratchpad for a specific session.

        Args:
            session_id: Session identifier.
        """
        if session_id in cls._scratchpads:
            cls._scratchpads[session_id] = []
