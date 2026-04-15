"""
Conversation Memory Module
Manages sliding-window conversation history for context-aware responses.
"""

import logging
from typing import Optional

from config.settings import MAX_MEMORY_TURNS

logger = logging.getLogger(__name__)


class ChatMemory:
    """Manages conversation history with a sliding window."""

    def __init__(self, max_turns: int = MAX_MEMORY_TURNS):
        """
        Initialize chat memory.

        Args:
            max_turns: Maximum number of conversation turns to retain.
        """
        self.max_turns = max_turns
        self.history: list[dict] = []

    def add_turn(self, role: str, content: str) -> None:
        """
        Add a conversation turn.

        Args:
            role: 'user' or 'assistant'.
            content: The message content.
        """
        self.history.append({
            "role": role,
            "content": content,
        })

        # Trim to sliding window (keep pairs intact)
        max_messages = self.max_turns * 2  # Each turn = user + assistant
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]

    def get_history(self) -> list[dict]:
        """Get the full conversation history within the window."""
        return self.history.copy()

    def format_history(self) -> str:
        """
        Format conversation history as a string for prompt inclusion.

        Returns:
            Formatted conversation history string.
        """
        if not self.history:
            return ""

        formatted_parts = []
        for turn in self.history:
            role = "User" if turn["role"] == "user" else "Assistant"
            # Truncate long messages in history to save context window
            content = turn["content"]
            if len(content) > 500:
                content = content[:500] + "..."
            formatted_parts.append(f"{role}: {content}")

        return "\n".join(formatted_parts)

    def clear(self) -> None:
        """Clear all conversation history."""
        self.history = []
        logger.info("Chat memory cleared")

    @property
    def turn_count(self) -> int:
        """Number of complete conversation turns (user+assistant pairs)."""
        return len(self.history) // 2

    @property
    def is_empty(self) -> bool:
        """Whether the memory is empty."""
        return len(self.history) == 0
