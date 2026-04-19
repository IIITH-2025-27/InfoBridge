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
    GEMINI_FALLBACK_MODELS,
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

    @staticmethod
    def _is_model_not_found(error_text: str) -> bool:
        return "404" in error_text and "not found" in error_text.lower()

    @staticmethod
    def _is_quota_error(error_text: str) -> bool:
        return "429" in error_text or "RESOURCE_EXHAUSTED" in error_text

    def _candidate_models(self) -> list[str]:
        models = [GEMINI_MODEL, *GEMINI_FALLBACK_MODELS]
        # Preserve order while removing duplicates.
        return list(dict.fromkeys([model for model in models if model]))

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
        config = types.GenerateContentConfig(
            temperature=GEMINI_TEMPERATURE,
            max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
            top_p=0.95,
        )

        if system_instruction:
            config.system_instruction = system_instruction

        model_not_found_seen = False
        quota_seen = False
        last_error_text = ""

        for model_name in self._candidate_models():
            try:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=config,
                )
                if response and response.text:
                    if model_name != GEMINI_MODEL:
                        logger.info("Gemini response generated via fallback model: %s", model_name)
                    return response.text.strip()

                logger.warning("Empty response from Gemini API using model: %s", model_name)
            except Exception as e:
                error_text = str(e)
                last_error_text = error_text

                if self._is_model_not_found(error_text):
                    model_not_found_seen = True
                    logger.warning("Gemini model not found: %s", model_name)
                    continue

                if self._is_quota_error(error_text):
                    quota_seen = True
                    logger.warning("Gemini quota exceeded for model %s: %s", model_name, error_text)
                    continue

                logger.error("Gemini API error for model %s: %s", model_name, error_text)

        if quota_seen:
            return (
                "I cannot generate a response right now because the Gemini API quota is exhausted. "
                "Please wait and try again, or use a different API key/project with available quota."
            )

        if model_not_found_seen:
            return (
                "I encountered a model configuration issue. "
                "Please set GEMINI_MODEL or GEMINI_FALLBACK_MODELS in your .env to supported models."
            )

        if last_error_text:
            return (
                "I encountered an error while generating a response. "
                f"Please try again. (Error: {last_error_text[:100]})"
            )

        return "I apologize, but I couldn't generate a response. Please try rephrasing your question."
