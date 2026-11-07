"""
LLM Client (provider-agnostic: Gemini by default, Groq optional)

All LLM calls in the project go through llmClient.generate(). It returns the answer
text, or one of the LLM_*_ERROR messages below when every model failed. Callers
should use is_llm_error() to detect those messages instead of matching text.

Gemini is called through its REST API with `requests` (already installed), so no
extra SDK is needed: paste a Gemini key into .env and it works.
"""

import logging
import random
import re
import time
from typing import Optional

import requests

from config.settings import (
    API_KEY,
    LLM_PROVIDER,
    MODEL,
    FALLBACK_MODELS,
    TEMPERATURE,
    MAX_OUTPUT_TOKENS,
    LLM_TIMEOUT_SECONDS,
    LLM_MAX_RETRIES,
    LLM_MAX_RETRY_WAIT_SECONDS,
    GEMINI_THINKING_LEVEL,
)

logger = logging.getLogger(__name__)

# ─── Error messages returned (not raised) by llmClient.generate ──────────────
LLM_QUOTA_ERROR = "Quota exceeded. Try again later."
LLM_AUTH_ERROR = "LLM API key is invalid or not permitted. Check GEMINI_API_KEY in .env."
LLM_MODEL_ERROR = "Model not found. Check MODEL / FALLBACK_MODELS in .env."
LLM_FAILED_ERROR = "Failed to generate response."
LLM_ERROR_PREFIX = "Error: "

_FIXED_ERRORS = {LLM_QUOTA_ERROR, LLM_AUTH_ERROR, LLM_MODEL_ERROR, LLM_FAILED_ERROR}

# HTTP statuses worth retrying on the same model before moving to the next one
_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


def is_llm_error(text: Optional[str]) -> bool:
    """True if `text` is an error message from llmClient.generate, not a real answer."""
    if not text or not text.strip():
        return True
    return text in _FIXED_ERRORS or text.startswith(LLM_ERROR_PREFIX)


class LLMRequestError(Exception):
    """A failed request to the provider, with the HTTP status when known."""

    def __init__(self, message: str, status: Optional[int] = None, retry_after: Optional[float] = None):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


# ─── Providers ───────────────────────────────────────────────────────────────

class _GeminiBackend:
    """Google Gemini via the generateContent REST endpoint."""

    URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        self.send_thinking_config = bool(GEMINI_THINKING_LEVEL)

    def complete(
        self,
        model: str,
        prompt: str,
        system_instruction: Optional[str],
        json_mode: bool,
        temperature: float,
        max_tokens: int,
    ) -> str:
        generation_config = {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        }
        if json_mode:
            generation_config["responseMimeType"] = "application/json"
        if self.send_thinking_config:
            generation_config["thinkingConfig"] = {"thinkingLevel": GEMINI_THINKING_LEVEL}

        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": generation_config,
        }
        if system_instruction:
            body["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        try:
            response = self.session.post(
                self.URL.format(model=model),
                headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
                json=body,
                timeout=LLM_TIMEOUT_SECONDS,
            )
        except requests.RequestException as exc:
            raise LLMRequestError(f"Network error: {exc}", status=None) from exc

        if response.status_code != 200:
            message = self._error_message(response)

            # Some models reject thinkingConfig; drop it once and retry.
            if response.status_code == 400 and self.send_thinking_config and "thinking" in message.lower():
                logger.warning("Model '%s' rejected thinkingConfig; retrying without it.", model)
                self.send_thinking_config = False
                return self.complete(model, prompt, system_instruction, json_mode, temperature, max_tokens)

            raise LLMRequestError(
                message,
                status=response.status_code,
                retry_after=self._retry_after(response),
            )

        data = response.json()
        candidates = data.get("candidates") or []
        if not candidates:
            block_reason = (data.get("promptFeedback") or {}).get("blockReason")
            raise LLMRequestError(f"Response blocked ({block_reason})" if block_reason else "Empty response")

        candidate = candidates[0]
        parts = (candidate.get("content") or {}).get("parts") or []
        # Skip "thought" parts if the model returns its reasoning separately.
        text = "".join(part.get("text", "") for part in parts if not part.get("thought"))

        if not text.strip():
            raise LLMRequestError(f"Empty response (finishReason={candidate.get('finishReason')})")

        return text.strip()

    @staticmethod
    def _error_message(response: requests.Response) -> str:
        try:
            return response.json().get("error", {}).get("message", "") or response.text[:300]
        except ValueError:
            return response.text[:300]

    @staticmethod
    def _retry_after(response: requests.Response) -> Optional[float]:
        header = response.headers.get("Retry-After")
        if header:
            try:
                return float(header)
            except ValueError:
                pass
        # Gemini puts the wait time in error.details[].retryDelay, e.g. "17s"
        try:
            for detail in response.json().get("error", {}).get("details", []):
                delay = detail.get("retryDelay")
                if delay:
                    match = re.match(r"([\d.]+)s", delay)
                    if match:
                        return float(match.group(1))
        except (ValueError, AttributeError):
            pass
        return None


class _GroqBackend:
    """Groq chat completions (kept so LLM_PROVIDER=groq still works)."""

    def __init__(self, api_key: str):
        from groq import Groq

        self.client = Groq(api_key=api_key, timeout=LLM_TIMEOUT_SECONDS, max_retries=0)

    def complete(
        self,
        model: str,
        prompt: str,
        system_instruction: Optional[str],
        json_mode: bool,
        temperature: float,
        max_tokens: int,
    ) -> str:
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        kwargs = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
        except Exception as exc:
            raise LLMRequestError(str(exc), status=getattr(exc, "status_code", None)) from exc

        if not response or not response.choices or not response.choices[0].message.content:
            raise LLMRequestError("Empty response")

        return response.choices[0].message.content.strip()


# ─── Public client ───────────────────────────────────────────────────────────

class llmClient:

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return

        if not API_KEY:
            raise ValueError("GEMINI_API_KEY not set in .env")

        if LLM_PROVIDER == "groq":
            self.backend = _GroqBackend(API_KEY)
        elif LLM_PROVIDER == "gemini":
            self.backend = _GeminiBackend(API_KEY)
        else:
            raise ValueError(f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. Use 'gemini' or 'groq'.")

        self._initialized = True
        logger.info(f"LLM initialized: provider={LLM_PROVIDER}, model={MODEL}")

    def _candidate_models(self):
        return list(dict.fromkeys([MODEL, *FALLBACK_MODELS]))

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        json_mode: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> str:
        """
        Generate text, trying the primary model then each fallback.

        Args:
            prompt: User prompt.
            system_instruction: Optional system prompt.
            json_mode: Ask the model to return a JSON object.
            temperature: Overrides TEMPERATURE (e.g. 0 for the eval judge).
            max_tokens: Overrides MAX_OUTPUT_TOKENS.
            model: Use only this model (e.g. a separate judge model in eval/).

        Returns:
            The answer text, or an LLM_*_ERROR message (check with is_llm_error).
        """
        temperature = TEMPERATURE if temperature is None else temperature
        max_tokens = MAX_OUTPUT_TOKENS if max_tokens is None else max_tokens

        last_error = ""
        quota_error = False
        model_not_found = False
        last_status_was_not_found = False

        for model_name in ([model] if model else self._candidate_models()):
            for attempt in range(LLM_MAX_RETRIES + 1):
                try:
                    text = self.backend.complete(
                        model=model_name,
                        prompt=prompt,
                        system_instruction=system_instruction,
                        json_mode=json_mode,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    if model_name != MODEL:
                        logger.info(f"Used fallback model: {model_name}")
                    return text

                except LLMRequestError as e:
                    last_error = str(e)
                    status = e.status
                    lower = last_error.lower()
                    last_status_was_not_found = False

                    # Wrong key: no other model will work either.
                    if status in (401, 403) or "api key not valid" in lower:
                        logger.error(f"LLM authentication failed: {last_error}")
                        return LLM_AUTH_ERROR

                    # Model removed / renamed / decommissioned → next model
                    if status == 404 or "decommission" in lower or "no longer available" in lower:
                        model_not_found = True
                        last_status_was_not_found = True
                        logger.warning(f"Model unavailable: {model_name} ({last_error[:120]})")
                        break

                    if status in _RETRYABLE_STATUSES:
                        if status == 429:
                            quota_error = True

                        wait = e.retry_after if e.retry_after is not None else (2 ** attempt) + random.random()
                        if attempt < LLM_MAX_RETRIES and wait <= LLM_MAX_RETRY_WAIT_SECONDS:
                            logger.warning(
                                f"LLM {status} on {model_name}; retrying in {wait:.1f}s "
                                f"(attempt {attempt + 1}/{LLM_MAX_RETRIES})"
                            )
                            time.sleep(wait)
                            continue

                        logger.warning(f"LLM {status} on {model_name}; moving to next model")
                        break

                    logger.error(f"LLM error ({model_name}): {last_error}")
                    break

        if quota_error:
            return LLM_QUOTA_ERROR

        if model_not_found and last_status_was_not_found:
            return LLM_MODEL_ERROR

        if last_error:
            return f"{LLM_ERROR_PREFIX}{last_error[:200]}"

        return LLM_FAILED_ERROR
