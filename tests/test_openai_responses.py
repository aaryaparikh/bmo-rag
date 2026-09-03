from __future__ import annotations

import json

from bmo_rag.generation.openai_responses import OpenAIResponsesClient


def test_responses_client_uses_gpt5_and_does_not_store(monkeypatch) -> None:
    client = OpenAIResponsesClient(api_key="test-key")
    captured: dict = {}

    class Response:
        status_code = 200

        @staticmethod
        def json() -> dict:
            return {
                "id": "resp-test",
                "model": "gpt-5",
                "status": "completed",
                "usage": {
                    "input_tokens": 12,
                    "output_tokens": 7,
                    "total_tokens": 19,
                    "input_tokens_details": {"cached_tokens": 3},
                    "output_tokens_details": {"reasoning_tokens": 2},
                },
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "Grounded answer [S1]."}],
                    }
                ],
            }

    def fake_post(url: str, **kwargs: object) -> Response:
        captured.update(url=url, **kwargs)
        return Response()

    monkeypatch.setattr(client.session, "post", fake_post)

    telemetry = []
    assert client.text(
        instructions="Ground answers", input_text="Question", telemetry=telemetry.append
    ) == (
        "Grounded answer [S1]."
    )
    assert captured["url"] == "https://api.openai.com/v1/responses"
    assert captured["json"]["model"] == "gpt-5"
    assert captured["json"]["store"] is False
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert telemetry[0].input_tokens == 12
    assert telemetry[0].output_tokens == 7
    assert telemetry[0].cached_input_tokens == 3
    assert telemetry[0].reasoning_tokens == 2


def test_responses_client_parses_structured_output(monkeypatch) -> None:
    client = OpenAIResponsesClient(api_key="test-key")

    class Response:
        status_code = 200

        @staticmethod
        def json() -> dict:
            return {
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": '{"query":"BMO Q2"}'}],
                    }
                ],
            }

    monkeypatch.setattr(client.session, "post", lambda *args, **kwargs: Response())
    result = client.structured(
        instructions="Rewrite",
        input_text="What about Q2?",
        schema_name="query",
        schema={"type": "object"},
    )
    assert result == {"query": "BMO Q2"}


def test_responses_client_streams_text_and_records_telemetry(monkeypatch) -> None:
    client = OpenAIResponsesClient(api_key="test-key")
    captured: dict = {}
    completed = {
        "id": "resp-stream",
        "model": "gpt-5",
        "status": "completed",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 4,
            "total_tokens": 14,
            "output_tokens_details": {"reasoning_tokens": 2},
        },
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "Hello world"}],
            }
        ],
    }

    class Response:
        status_code = 200

        def iter_lines(self, **kwargs: object) -> list[str]:
            return [
                'data: {"type":"response.output_text.delta","delta":"Hello "}',
                "",
                'data: {"type":"response.output_text.delta","delta":"world"}',
                "",
                f'data: {json.dumps({"type": "response.completed", "response": completed})}',
                "",
            ]

        def close(self) -> None:
            pass

    def fake_post(url: str, **kwargs: object) -> Response:
        captured.update(url=url, **kwargs)
        return Response()

    monkeypatch.setattr(client.session, "post", fake_post)
    deltas: list[str] = []
    telemetry = []

    result = client.text(
        instructions="Answer",
        input_text="Question",
        on_text_delta=deltas.append,
        telemetry=telemetry.append,
    )

    assert result == "Hello world"
    assert deltas == ["Hello ", "world"]
    assert captured["json"]["stream"] is True
    assert captured["stream"] is True
    assert telemetry[0].total_tokens == 14
