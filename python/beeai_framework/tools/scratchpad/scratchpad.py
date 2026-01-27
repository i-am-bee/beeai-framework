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

import asyncio
import logging
import re
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
    """Tool for managing agent scratchpad (working memory).

    Supports two types of content:
    1. Key-Value Pairs: Automatically consolidated into a single entry
       - Format: "key: value, another_key: another_value"
       - Duplicate keys are updated with latest values
       - Results in ONE entry containing all key-value pairs

    2. Plain Text: Appended as separate entries
       - Format: Any text without colons
       - Each write creates a new list entry

    This design allows structured state management (key-value) while
    preserving free-form notes (plain text) as separate items.
    """

    _scratchpads: ClassVar[dict[str, list]] = {}
    _lock: ClassVar[asyncio.Lock] = asyncio.Lock()

    def __init__(self) -> None:
        """Initialize scratchpad tool."""
        super().__init__()
        self.middlewares = []
        # Store the session_id once it's determined from context
        # This ensures the same session is used across all calls
        self._cached_session_id: str | None = None

    @staticmethod
    def _ensure_session(session_id: str) -> None:
        """Ensure a session exists in scratchpads."""
        if session_id not in ScratchpadTool._scratchpads:
            ScratchpadTool._scratchpads[session_id] = []

    def _get_session_id(self, context: RunContext | None = None) -> str:
        """Extract session ID from context.

        Caches the session ID on first call to ensure the same session
        is used across all tool calls for this tool instance.

        Args:
            context: Run context to extract session identifier from.

        Returns:
            Session ID string for data isolation.

        Raises:
            ValueError: If no valid session ID can be extracted from context.
        """
        # Return cached session ID if we already determined it
        if self._cached_session_id:
            return self._cached_session_id

        if not context:
            raise ValueError(
                "Scratchpad requires RunContext with a valid session identifier. "
                "No context provided."
            )

        # Try different context attributes in order of preference
        session_id = None

        # run_id: Should persist across tool calls in the same agent run
        if hasattr(context, "run_id") and context.run_id:
            session_id = str(context.run_id)
            logger.debug(f"Using run_id as session: {session_id}")

        # conversation_id: If available, persists across the conversation
        elif hasattr(context, "conversation_id") and context.conversation_id:
            session_id = str(context.conversation_id)
            logger.debug(f"Using conversation_id as session: {session_id}")

        # agent_id: If available, unique per agent instance
        elif hasattr(context, "agent_id") and context.agent_id:
            session_id = str(context.agent_id)
            logger.debug(f"Using agent_id as session: {session_id}")

        # No valid session ID found - raise error
        if not session_id:
            raise ValueError(
                "Scratchpad requires RunContext with a valid session identifier "
                "(run_id, conversation_id, or agent_id). None found in context."
            )

        # Cache the session ID for future calls
        self._cached_session_id = session_id
        logger.info(f"Scratchpad session initialized: {session_id}")
        return session_id

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

        Uses regex to correctly handle values containing commas, which prevents
        incorrectly splitting "item: milk, bread, eggs" into separate entries.

        Format Examples:
        - Simple: "key: value" → {"key": "value"}
        - Multiple: "key1: val1, key2: val2" → {"key1": "val1", "key2": "val2"}
        - Comma in value: "item: milk, bread, key2: val2" → {"item": "milk, bread", "key2": "val2"}
        - Hyphenated keys: "Content-Type: json, user-id: 123" → {"Content-Type": "json", "user-id": "123"}

        Implementation Note:
        A simple split-by-comma approach fails when values contain commas.
        The regex pattern works by:
        1. Matching any characters except ':' as the key: ([^:]+)
        2. Matching the colon separator: :
        3. Capturing everything until the next "key:" pattern or end: (.*?)(?=...)

        This ensures commas within values are preserved while correctly
        identifying multiple key-value pairs separated by commas.

        Args:
            content: Content string to parse.

        Returns:
            Dictionary of key-value pairs.
        """
        pairs = {}

        # Regex breakdown:
        # ([^:]+)           - Capture key (any chars except colon)
        # :\s*              - Match colon and optional whitespace
        # (.*?)             - Capture value (non-greedy)
        # (?=               - Lookahead (doesn't consume characters):
        #   \s*,\s*[^:]+:   - Next key-value pair (comma, then key:)
        #   |               - OR
        #   \s*$            - End of string
        # )
        pattern = re.compile(r"([^:]+):\s*(.*?)(?=\s*,\s*[^:]+:|\s*$)")

        for match in pattern.finditer(content):
            key = match.group(1).strip()
            value = match.group(2).strip()
            if key and value:
                pairs[key] = value

        return pairs

    @staticmethod
    def _merge_entries(entries: list, new_pairs: dict) -> list:
        """Merge new key-value pairs into existing entries.

        IMPORTANT BEHAVIOR:
        This method consolidates ALL key-value pairs (existing + new) into a
        SINGLE entry. This means the returned list will contain at most ONE
        consolidated entry, not multiple separate entries.

        Design Rationale:
        - Key-value pairs represent structured state that should be merged
        - Each key appears only once with its latest value
        - Prevents duplicate keys and maintains a single source of truth
        - Example: Writing "city: Boston" then "date: 2025-01-28" results in
          ONE entry: "city: Boston, date: 2025-01-28"

        This is different from non-key-value entries (plain text) which are
        appended as separate list items.

        Args:
            entries: List of existing scratchpad entries.
            new_pairs: Dictionary of new key-value pairs to merge.

        Returns:
            Updated list with a SINGLE consolidated entry containing all
            key-value pairs (old + new), with new values overriding old
            values for duplicate keys. Returns empty list if no valid pairs.
        """
        # Parse all existing entries into a single dict
        consolidated = {}
        for entry in entries:
            pairs = ScratchpadTool._parse_key_value_pairs(entry)
            consolidated.update(pairs)

        # Merge new pairs (new values override old ones for duplicate keys)
        consolidated.update(new_pairs)

        # Convert back to entry format
        if consolidated:
            # Create a single consolidated entry containing all pairs
            entry_str = ", ".join(f"{k}: {v}" for k, v in consolidated.items())
            return [entry_str]
        return []

    def _write_scratchpad(self, entry: str, session_id: str) -> str:
        """Add or update entry in the scratchpad.

        Behavior depends on entry format:

        1. Key-Value Pairs (contains ":"):
           - Parsed and merged with existing key-value pairs
           - Results in a SINGLE consolidated entry
           - New values override old values for duplicate keys
           - Example: Writing "city: Boston" then "date: 2025-01-28" creates
             ONE entry: "city: Boston, date: 2025-01-28"

        2. Plain Text (no ":"):
           - Appended as a new separate entry
           - Multiple plain text entries can exist

        This design ensures structured state (key-value) remains consolidated
        while allowing free-form notes (plain text) to accumulate.

        Args:
            entry: Content to add/update.
            session_id: Session identifier.

        Returns:
            Success message describing the action taken.
        """
        entries = self._get_entries(session_id)
        new_pairs = self._parse_key_value_pairs(entry)

        if new_pairs:
            # Merge with existing entries
            entries[:] = self._merge_entries(entries, new_pairs)
            result = (
                f"Updated scratchpad to: {entries[0]}"
                if entries
                else "Scratchpad updated with no content."
            )
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
        session_id = self._get_session_id(context)
        operation = input.operation.lower().strip()
        content = input.content

        logger.info(
            f"ScratchpadTool[{session_id}]: operation='{operation}', "
            f"content='{content}'"
        )

        result = None
        async with ScratchpadTool._lock:
            self._ensure_session(session_id)

            handlers = {
                "read": lambda: self._read_scratchpad(session_id),
                "write": lambda: (
                    self._write_scratchpad(content, session_id)
                    if content
                    else self._raise_input_validation_error(
                        "'write' operation requires 'content' parameter."
                    )
                ),
                "append": lambda: (
                    self._append_scratchpad(content, session_id)
                    if content
                    else self._raise_input_validation_error(
                        "'append' operation requires 'content' parameter."
                    )
                ),
                "clear": lambda: self._clear_scratchpad(session_id),
            }

            handler = handlers.get(operation)
            if handler:
                result = handler()

        if result is not None:
            return StringToolOutput(result=result)

        error_msg = (
            f"Unknown operation: {operation}. "
            "Use 'read', 'write', 'append', or 'clear'."
        )
        return StringToolOutput(result=error_msg)

    def _raise_input_validation_error(self, message: str) -> None:
        """Raise a ToolInputValidationError with the given message.

        Args:
            message: Error message to include in the exception.

        Raises:
            ToolInputValidationError: Always raised with the provided message.
        """
        from beeai_framework.tools import ToolInputValidationError

        raise ToolInputValidationError(message)

    @classmethod
    def get_scratchpad_for_session(cls, session_id: str) -> list:
        """Get scratchpad entries for a specific session.

        Args:
            session_id: Session identifier.

        Returns:
            List of scratchpad entries.
        """
        return cls._scratchpads.get(session_id, []).copy()

    @classmethod
    async def clear_session(cls, session_id: str) -> None:
        """Clear scratchpad for a specific session.

        Args:
            session_id: Session identifier.
        """
        async with cls._lock:
            if session_id in cls._scratchpads:
                cls._scratchpads[session_id] = []
