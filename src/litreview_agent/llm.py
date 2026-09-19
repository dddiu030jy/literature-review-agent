from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from typing import Any


class LLMError(RuntimeError):
    pass


@dataclass
class LLMCall:
    trace_id: str
    stage: str
    prompt_version: str
    model: str
    input_chars: int
    output_chars: int
    attempt: int
    duration_seconds: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    temperature: float
    top_p: float
    max_tokens: int
    response_request_id: str


class OpenAICompatibleClient:
    """Minimal Chat Completions client with JSON repair and retry."""

    def __init__(
        self,
        *,
        mode: str = "auto",
        model: str,
        base_url: str,
        timeout: int = 60,
        top_p: float = 1.0,
        max_tokens: int = 4096,
        retries: int = 3,
    ) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.mode = mode
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.retries = retries
        self.calls: list[LLMCall] = []
        self.failed_calls = 0
        self.json_repair_attempts = 0

    @property
    def enabled(self) -> bool:
        if self.mode == "heuristic":
            return False
        if self.mode == "openai" and not self.api_key:
            raise LLMError("llm_mode=openai requires OPENAI_API_KEY")
        return bool(self.api_key)

    def complete(
        self,
        *,
        stage: str,
        prompt_version: str,
        system: str,
        user: str,
        json_mode: bool = False,
        temperature: float = 0.2,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        if not self.enabled:
            raise LLMError("LLM is disabled; use a stage-specific heuristic fallback")
        url = self.base_url
        if not url.endswith("/chat/completions"):
            url += "/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": temperature,
            "top_p": self.top_p if top_p is None else top_p,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        last_error: Exception | None = None
        trace_id = str(uuid.uuid4())
        for attempt in range(1, self.retries + 1):
            request = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                method="POST",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "LitReviewAgent/1.0",
                },
            )
            call_started = time.perf_counter()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                    response_request_id = response.headers.get("x-request-id", "")
                text = body["choices"][0]["message"]["content"]
                usage = body.get("usage") or {}
                self.calls.append(
                    LLMCall(
                        trace_id=trace_id,
                        stage=stage,
                        prompt_version=prompt_version,
                        model=self.model,
                        input_chars=len(system) + len(user),
                        output_chars=len(text),
                        attempt=attempt,
                        duration_seconds=round(time.perf_counter() - call_started, 4),
                        prompt_tokens=_optional_int(usage.get("prompt_tokens")),
                        completion_tokens=_optional_int(usage.get("completion_tokens")),
                        total_tokens=_optional_int(usage.get("total_tokens")),
                        temperature=temperature,
                        top_p=float(payload["top_p"]),
                        max_tokens=int(payload["max_tokens"]),
                        response_request_id=response_request_id,
                    )
                )
                return text
            except (urllib.error.URLError, urllib.error.HTTPError, KeyError, ValueError) as exc:
                last_error = exc
                self.failed_calls += 1
                if attempt < self.retries:
                    time.sleep(1.2 * attempt)
        raise LLMError(
            f"LLM call failed after {self.retries} attempts: {last_error}"
        )

    def complete_json(self, **kwargs: Any) -> dict[str, Any]:
        text = self.complete(json_mode=True, **kwargs)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            self.json_repair_attempts += 1
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                raise LLMError("LLM did not return a JSON object")
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError as exc:
                raise LLMError(f"Could not repair LLM JSON: {exc}") from exc


def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
