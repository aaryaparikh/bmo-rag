from __future__ import annotations

from bmo_rag.cli import _run_chat_session
from bmo_rag.pipeline.chat import ChatAnswer


class FakeChatbot:
    def __init__(self, *, emit_deltas: bool = True) -> None:
        self.questions: list[str] = []
        self.emit_deltas = emit_deltas

    def answer(self, question: str, **kwargs: object) -> ChatAnswer:
        self.questions.append(question)
        on_text_delta = kwargs.get("on_text_delta")
        if self.emit_deltas and callable(on_text_delta):
            on_text_delta(f"Answer: {question}")
        return ChatAnswer(
            answer=f"Answer: {question}",
            original_question=question,
            retrieval_query=question,
            sources=(),
            requested_sources=(),
            context_truncated=False,
        )


def test_chat_session_keeps_prompting_after_initial_question(monkeypatch, capsys) -> None:
    responses = iter(["follow up", "", "quit"])
    monkeypatch.setattr("bmo_rag.cli.typer.prompt", lambda _: next(responses))
    chatbot = FakeChatbot()

    _run_chat_session(chatbot, "first question", loop=True, stream=True)  # type: ignore[arg-type]

    assert chatbot.questions == ["first question", "follow up"]
    output = capsys.readouterr().out
    assert "Memory-enabled BMO chat" in output
    assert output.count("Answer: first question") == 1


def test_chat_session_can_still_run_one_question(capsys) -> None:
    chatbot = FakeChatbot()

    _run_chat_session(chatbot, "one shot", loop=False, stream=False)  # type: ignore[arg-type]

    assert chatbot.questions == ["one shot"]
    assert "Memory-enabled BMO chat" not in capsys.readouterr().out


def test_streamed_session_prints_fallback_when_no_deltas_arrive(capsys) -> None:
    chatbot = FakeChatbot(emit_deltas=False)

    _run_chat_session(chatbot, "no evidence", loop=False, stream=True)  # type: ignore[arg-type]

    assert capsys.readouterr().out.count("Answer: no evidence") == 1
