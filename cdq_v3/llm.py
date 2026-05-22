from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .cache import PromptCache

DEFAULT_ATTR_PANEL = {
    "N1": "Yes",
    "N2": "No",
    "N3": "Uncertain",
    "N4": "No",
    "N5": "Yes",
    "F1": "Yes",
    "F2": "No",
    "F3": "Uncertain",
    "F4": "Yes",
    "F5": "No",
}


@dataclass
class LLMResponse:
    text: str
    raw: Any | None = None


class LLM:
    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
        stop: Optional[List[str]] = None,
    ) -> LLMResponse:
        raise NotImplementedError


class CachedLLM(LLM):
    """Adds sqlite caching on top of another LLM."""

    def __init__(self, base_llm: LLM, cache: PromptCache):
        self.base_llm = base_llm
        self.cache = cache

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
        stop: Optional[List[str]] = None,
    ) -> LLMResponse:
        payload: Dict[str, Any] = {
            "system": system,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stop": stop or [],
            "model": getattr(self.base_llm, "model", "unknown"),
        }
        cached = self.cache.get(payload)
        if cached:
            return LLMResponse(text=cached["text"])
        response = self.base_llm.generate(
            prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
        )
        self.cache.set(payload, {"text": response.text})
        return response


class OpenAIChatLLM(LLM):
    """Thin wrapper around the official OpenAI SDK."""

    def __init__(
        self,
        model: str,
        *,
        api_key: Optional[str] = None,
        api_key_env: str = "OPENAI_API_KEY",
        organization: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 60,
    ):
        self.model = model
        self.api_key = api_key or os.getenv(api_key_env)
        self.api_key_env = api_key_env
        self.organization = organization
        self.base_url = base_url
        self.timeout = timeout

    def _use_responses_api(self) -> bool:
        lowered = (self.model or "").lower()
        return lowered.startswith(("gpt-5", "o1", "o3"))

    @staticmethod
    def _wants_json(prompt: str, system: Optional[str]) -> bool:
        text = f"{system or ''}\n{prompt}".lower()
        return "json" in text

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
        stop: Optional[List[str]] = None,
    ) -> LLMResponse:
        from openai import APITimeoutError, OpenAI, RateLimitError  # type: ignore
        import httpx
        import time

        client = OpenAI(
            api_key=self.api_key,
            organization=self.organization,
            timeout=self.timeout,
            base_url=self.base_url,
        )

        def should_retry(exc: Exception) -> bool:
            if isinstance(exc, (APITimeoutError, RateLimitError)):
                return True
            if isinstance(exc, httpx.TimeoutException):
                return True
            status = getattr(exc, "status_code", None)
            if status == 429:
                return True
            response = getattr(exc, "response", None)
            if response is not None and getattr(response, "status_code", None) == 429:
                return True
            return False

        def call_with_retry(fn):
            attempts = 3
            for attempt in range(attempts):
                try:
                    return fn()
                except Exception as exc:
                    if attempt >= attempts - 1 or not should_retry(exc):
                        raise
                    time.sleep(2**attempt)
        if self._use_responses_api():
            input_blocks = []
            if system:
                input_blocks.append(
                    {
                        "role": "system",
                        "content": [{"type": "input_text", "text": system}],
                    }
                )
            input_blocks.append(
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                }
            )
            params = {
                "model": self.model,
                "input": input_blocks,
                "max_output_tokens": max_tokens,
            }
            # GPT-5.* does not support temperature in Responses API.
            if not (self.model or "").lower().startswith("gpt-5"):
                params["temperature"] = temperature
            if (self.model or "").lower().startswith("gpt-5") and self._wants_json(prompt, system):
                params["response_format"] = {"type": "json_object"}
            response = call_with_retry(lambda: client.responses.create(**params))
            text = self._extract_response_text(response)
            # GPT-5 responses can return reasoning-only items with empty output. Fallback to chat completions.
            if not text and (self.model or "").lower().startswith("gpt-5"):
                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": prompt})
                kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "stop": stop,
                    "max_completion_tokens": max_tokens,
                }
                chat_response = call_with_retry(lambda: client.chat.completions.create(**kwargs))
                text = chat_response.choices[0].message.content or ""
                return LLMResponse(text=text, raw={"responses": response, "chat": chat_response})
            return LLMResponse(text=text, raw=response)

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stop": stop,
        }
        lowered = (self.model or "").lower()
        # GPT-4.x and earlier use max_tokens; GPT-5.x chat completions expect max_completion_tokens.
        if lowered.startswith("gpt-5"):
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens
        response = call_with_retry(lambda: client.chat.completions.create(**kwargs))
        text = response.choices[0].message.content or ""
        return LLMResponse(text=text, raw=response)

    @staticmethod
    def _extract_response_text(resp: Any) -> str:
        output_text = getattr(resp, "output_text", None)
        if output_text:
            if isinstance(output_text, list):
                return "\n".join(str(t) for t in output_text).strip()
            return str(output_text).strip()
        output = getattr(resp, "output", None)
        if output:
            chunks = []
            for item in output:
                if isinstance(item, dict):
                    content = item.get("content")
                    text = item.get("text") or item.get("output_text") or item.get("value")
                    if content is None and text:
                        chunks.append(str(text))
                        continue
                else:
                    content = getattr(item, "content", None)
                    text = getattr(item, "text", None) or getattr(item, "output_text", None) or getattr(item, "value", None)
                    if content is None and text:
                        chunks.append(str(text))
                        continue
                if not content:
                    continue
                if isinstance(content, str):
                    chunks.append(content)
                    continue
                for block in content:
                    if isinstance(block, dict):
                        text = block.get("text") or block.get("output_text") or block.get("value")
                        if isinstance(text, list):
                            chunks.extend(str(t) for t in text)
                        elif text:
                            chunks.append(str(text))
                    else:
                        value = (
                            getattr(block, "text", None)
                            or getattr(block, "output_text", None)
                            or getattr(block, "value", None)
                        )
                        if value:
                            chunks.append(str(value))
            if chunks:
                text = "\n".join(chunks)
                return re.sub(r"\s+", " ", text).strip()
        return re.sub(r"\s+", " ", str(resp)).strip()


class RuleBasedMockLLM(LLM):
    """Deterministic mock that returns JSON payloads matching prompts."""

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
        stop: Optional[List[str]] = None,
    ) -> LLMResponse:
        lower = prompt.lower()
        if "\"questions\"" in lower or "questions" in lower:
            data = {
                "questions": [
                    "What evidence is missing to confirm the main mechanism?",
                    "Which assumptions fail under boundary shifts?",
                    "How could measurement choices alter feasibility?",
                ]
            }
            return LLMResponse(text=json.dumps(data))
        if "\"answer\"" in lower and "\"tag\"" in lower:
            data = {"answer": "Mock answer grounded in provided snippets.", "tag": "Partial"}
            return LLMResponse(text=json.dumps(data))
        if "\"entailment\"" in lower and "\"tag\"" in lower:
            data = {"tag": "Unknown", "entailment": "NotEntailed"}
            return LLMResponse(text=json.dumps(data))
        if "\"relation\"" in lower:
            return LLMResponse(text=json.dumps({"relation": "Unrelated"}))
        if "return json with each attribute key" in lower or "attributes" in lower:
            panel = {k: DEFAULT_ATTR_PANEL.get(k, "Uncertain") for k in DEFAULT_ATTR_PANEL}
            return LLMResponse(text=json.dumps(panel))
        return LLMResponse(text=json.dumps({"mock": "response"}))


class MockAnthropicLLM(RuleBasedMockLLM):
    """Concrete mock for non-OpenAI provider; inherits rule-based behavior."""

    pass


class MockGroqLLM(RuleBasedMockLLM):
    """Another deterministic placeholder."""

    pass
