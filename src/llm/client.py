"""
LLM Client (GROQ VERSION - GENERIC)
"""

import logging
from typing import Optional

from groq import Groq

from config.settings import (
    API_KEY,
    MODEL,
    FALLBACK_MODELS,
    TEMPERATURE,
    MAX_OUTPUT_TOKENS,
)

logger = logging.getLogger(__name__)


class llmClient:

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not API_KEY:
            raise ValueError("API_KEY not set in .env")

        self.client = Groq(api_key=API_KEY)

        logger.info(f"LLM initialized with model: {MODEL}")

    def _candidate_models(self):
        return list(dict.fromkeys([MODEL, *FALLBACK_MODELS]))

    def generate(self, prompt: str, system_instruction: Optional[str] = None):

        last_error = ""
        model_not_found = False
        quota_error = False

        for model_name in self._candidate_models():
            try:

                messages = []

                if system_instruction:
                    messages.append({
                        "role": "system",
                        "content": system_instruction
                    })

                messages.append({
                    "role": "user",
                    "content": prompt
                })

                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_OUTPUT_TOKENS,
                )

                if response and response.choices:
                    if model_name != MODEL:
                        logger.info(f"Used fallback model: {model_name}")

                    return response.choices[0].message.content.strip()

            except Exception as e:
                error_text = str(e)
                last_error = error_text

                if "404" in error_text:
                    model_not_found = True
                    continue

                if "429" in error_text:
                    quota_error = True
                    continue

                logger.error(f"LLM error ({model_name}): {error_text}")

        if quota_error:
            return "Quota exceeded. Try again later."

        if model_not_found:
            return "Model not found. Check MODEL config."

        if last_error:
            return f"Error: {last_error[:100]}"

        return "Failed to generate response."