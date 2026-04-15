"""
Gemini LLM Client
Wrapper around Google GenAI SDK for Gemini API calls.
"""

import logging
from typing import Optional

from google import genai
from google.genai import types

from config.settings import (
    GOOGLE_API_KEY,
    GEMINI_MODEL,
    GEMINI_TEMPERATURE,
    GEMINI_MAX_OUTPUT_TOKENS,
)

logger = logging.getLogger(__name__)


class GeminiClient:
    """Client for Google Gemini API interactions."""

    _instance = None
    _client = None

    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Configure and initialize the Gemini client."""
        if GeminiClient._client is None:
            if not GOOGLE_API_KEY:
                raise ValueError(
                    "GOOGLE_API_KEY not set. "
                    "Create a .env file with GOOGLE_API_KEY=your_key_here"
                )

            GeminiClient._client = genai.Client(api_key=GOOGLE_API_KEY)
            logger.info(f"Gemini client initialized for model: {GEMINI_MODEL}")

    @property
    def client(self):
        return GeminiClient._client

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
    ) -> str:
        """
        Generate a response from Gemini.

        Args:
            prompt: The full prompt to send.
            system_instruction: Optional system-level instruction.

        Returns:
            Generated text response.
        """
        try:
            config = types.GenerateContentConfig(
                temperature=GEMINI_TEMPERATURE,
                max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
                top_p=0.95,
            )

            if system_instruction:
                config.system_instruction = system_instruction

            response = self.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=config,
            )

            if response and response.text:
                return response.text.strip()
            else:
                logger.warning("Empty response from Gemini API")
                return "I apologize, but I couldn't generate a response. Please try rephrasing your question."

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return f"I encountered an error while generating a response. Please try again. (Error: {str(e)[:100]})"
