"""Minimal OpenAI Responses API client used by the chat pipeline."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import requests


class OpenAIError(RuntimeError):
    """Raised when an OpenAI response cannot be produced or decoded."""


@dataclass(frozen=True)
class ResponseTelemetry:
    operation: str
    response_id: str | None
    model: str
    instructions: str
    input_text: str
    output_text: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_input_tokens: int
    reasoning_tokens: int


TelemetryCallback = Callable[[ResponseTelemetry], None]
TextDeltaCallback = Callable[[str], None]


class OpenAIResponsesClient:
    def __init__(
        self,
        *,
        model: str = "gpt-5",
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise OpenAIError(
                "OPENAI_API_KEY is not set. Put it in the environment or the local .env file."
            )
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()

    def text(
        self,
        *,
        instructions: str,
        input_text: str,
        max_output_tokens: int = 1200,
        reasoning_effort: str = "low",
        telemetry: TelemetryCallback | None = None,
        operation: str = "generation",
        on_text_delta: TextDeltaCallback | None = None,
    ) -> str:
        if on_text_delta is None:
            body = self._create(
                instructions=instructions,
                input_text=input_text,
                max_output_tokens=max_output_tokens,
                reasoning_effort=reasoning_effort,
            )
        else:
            body = self._create_stream(
                instructions=instructions,
                input_text=input_text,
                max_output_tokens=max_output_tokens,
                reasoning_effort=reasoning_effort,
                on_text_delta=on_text_delta,
            )
        output = self._output_text(body)
        if telemetry:
            telemetry(
                self._telemetry(
                    body,
                    operation=operation,
                    instructions=instructions,
                    input_text=input_text,
                    output_text=output,
                )
            )
        return output

    def structured(
        self,
        *,
        instructions: str,
        input_text: str,
        schema_name: str,
        schema: dict[str, Any],
        max_output_tokens: int = 700,
        reasoning_effort: str = "low",
        telemetry: TelemetryCallback | None = None,
        operation: str = "structured_output",
    ) -> dict[str, Any]:
        body = self._create(
            instructions=instructions,
            input_text=input_text,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
            text_format={
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": schema,
            },
        )
        output = self._output_text(body)
        if telemetry:
            telemetry(
                self._telemetry(
                    body,
                    operation=operation,
                    instructions=instructions,
                    input_text=input_text,
                    output_text=output,
                )
            )
        try:
            value = json.loads(output)
        except json.JSONDecodeError as exc:
            raise OpenAIError("OpenAI returned invalid structured JSON") from exc
        if not isinstance(value, dict):
            raise OpenAIError("OpenAI structured output was not an object")
        return value

    def _create(
        self,
        *,
        instructions: str,
        input_text: str,
        max_output_tokens: int,
        reasoning_effort: str,
        text_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = self._payload(
            instructions=instructions,
            input_text=input_text,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
            text_format=text_format,
        )
        return self._validate_response(self._post(payload).json())

    def _create_stream(
        self,
        *,
        instructions: str,
        input_text: str,
        max_output_tokens: int,
        reasoning_effort: str,
        on_text_delta: TextDeltaCallback,
    ) -> dict[str, Any]:
        payload = self._payload(
            instructions=instructions,
            input_text=input_text,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
        )
        payload["stream"] = True
        response = self._post(payload, stream=True)
        completed: dict[str, Any] | None = None
        try:
            for event in self._iter_sse_events(response):
                event_type = event.get("type")
                if event_type == "response.output_text.delta":
                    delta = str(event.get("delta") or "")
                    if delta:
                        on_text_delta(delta)
                elif event_type == "response.completed":
                    value = event.get("response")
                    if isinstance(value, dict):
                        completed = value
                elif event_type in {"response.failed", "response.incomplete"}:
                    value = event.get("response")
                    if isinstance(value, dict):
                        self._validate_response(value)
                    raise OpenAIError(f"OpenAI stream ended with {event_type}")
                elif event_type == "error":
                    raise OpenAIError(
                        f"OpenAI stream failed: {event.get('message') or 'unknown error'}"
                    )
        except requests.RequestException as exc:
            raise OpenAIError(f"OpenAI stream was interrupted: {exc}") from exc
        finally:
            response.close()
        if completed is None:
            raise OpenAIError("OpenAI stream ended without a completed response")
        return self._validate_response(completed)

    def _payload(
        self,
        *,
        instructions: str,
        input_text: str,
        max_output_tokens: int,
        reasoning_effort: str,
        text_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "instructions": instructions,
            "input": input_text,
            "max_output_tokens": max_output_tokens,
            "reasoning": {"effort": reasoning_effort},
            "store": False,
        }
        if text_format is not None:
            payload["text"] = {"format": text_format}
        return payload

    def _post(self, payload: dict[str, Any], *, stream: bool = False) -> requests.Response:
        response: requests.Response | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.post(
                    f"{self.base_url}/responses",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    stream=stream,
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                if attempt == self.max_retries:
                    raise OpenAIError(f"Cannot reach OpenAI: {exc}") from exc
                time.sleep(min(2**attempt, 10))
                continue
            if response.status_code < 400:
                return response
            retryable = response.status_code in {408, 409, 429} or response.status_code >= 500
            if not retryable or attempt == self.max_retries:
                raise OpenAIError(
                    f"OpenAI returned {response.status_code}: {response.text[:1000]}"
                )
            time.sleep(min(2**attempt, 10))
        raise AssertionError("unreachable")

    @staticmethod
    def _iter_sse_events(response: requests.Response) -> Iterator[dict[str, Any]]:
        data_lines: list[str] = []
        for raw_line in response.iter_lines(chunk_size=1, decode_unicode=True):
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            if line == "":
                if data_lines:
                    data = "\n".join(data_lines)
                    data_lines.clear()
                    if data != "[DONE]":
                        value = json.loads(data)
                        if isinstance(value, dict):
                            yield value
                continue
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
        if data_lines:
            value = json.loads("\n".join(data_lines))
            if isinstance(value, dict):
                yield value

    @staticmethod
    def _validate_response(result: dict[str, Any]) -> dict[str, Any]:
        if result.get("status") == "incomplete":
            reason = (result.get("incomplete_details") or {}).get("reason", "unknown")
            raise OpenAIError(f"OpenAI response was incomplete: {reason}")
        if result.get("status") == "failed":
            message = (result.get("error") or {}).get("message", "unknown error")
            raise OpenAIError(f"OpenAI response failed: {message}")
        return result

    @staticmethod
    def _output_text(response: dict[str, Any]) -> str:
        texts = [
            str(content.get("text", ""))
            for item in response.get("output", [])
            if item.get("type") == "message"
            for content in item.get("content", [])
            if content.get("type") == "output_text"
        ]
        output = "".join(texts).strip()
        if not output:
            raise OpenAIError("OpenAI response did not contain output text")
        return output

    def _telemetry(
        self,
        response: dict[str, Any],
        *,
        operation: str,
        instructions: str,
        input_text: str,
        output_text: str,
    ) -> ResponseTelemetry:
        usage = response.get("usage") or {}
        input_details = usage.get("input_tokens_details") or {}
        output_details = usage.get("output_tokens_details") or {}
        return ResponseTelemetry(
            operation=operation,
            response_id=str(response["id"]) if response.get("id") else None,
            model=str(response.get("model") or self.model),
            instructions=instructions,
            input_text=input_text,
            output_text=output_text,
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            total_tokens=int(usage.get("total_tokens") or 0),
            cached_input_tokens=int(input_details.get("cached_tokens") or 0),
            reasoning_tokens=int(output_details.get("reasoning_tokens") or 0),
        )
