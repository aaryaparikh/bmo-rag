from __future__ import annotations

from bmo_rag.retrieval.reranker import VllmReranker


def test_vllm_reranker_maps_scores_back_to_points(monkeypatch) -> None:
    reranker = VllmReranker()
    captured: dict = {}

    class Response:
        status_code = 200

        @staticmethod
        def json() -> dict:
            return {
                "results": [
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.2},
                ]
            }

    def fake_post(url: str, *, json: dict, timeout: float) -> Response:
        captured.update(url=url, body=json, timeout=timeout)
        return Response()

    monkeypatch.setattr(reranker.session, "post", fake_post)
    points = [
        {"score": 0.7, "payload": {"text": "first"}},
        {"score": 0.6, "payload": {"headings": ["Section"], "text": "second"}},
    ]

    result = reranker.rerank("question", points, top_k=2)

    assert [point["payload"]["text"] for point in result] == ["second", "first"]
    assert result[0]["rerank_score"] == 0.9
    assert result[0]["fusion_score"] == 0.6
    assert captured["url"] == "http://127.0.0.1:8001/rerank"
    assert captured["body"]["documents"][1] == "Section\n\nsecond"
