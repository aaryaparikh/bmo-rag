from __future__ import annotations

from bmo_rag.generation.memory import ConversationMemory


class FakeClient:
    def structured(self, **kwargs: object) -> dict:
        return {
            "standalone_query": "What was BMO U.S. Banking net income in Q2 2026?",
            "state": {
                "topic": "net income",
                "entity": "BMO",
                "reporting_period": "Q2 2026",
                "business_segment": "U.S. Banking",
                "constraints": "",
                "summary": "The user is comparing segment net income.",
            },
        }


def test_first_question_skips_rewrite() -> None:
    memory = ConversationMemory()
    assert memory.prepare_query("What was net income?", FakeClient()) == "What was net income?"


def test_follow_up_is_rewritten_and_state_is_updated() -> None:
    memory = ConversationMemory(max_turns=2)
    memory.remember("What was net income in Q2 2026?", "It was X [S1].")

    query = memory.prepare_query("What about U.S. Banking?", FakeClient())

    assert query == "What was BMO U.S. Banking net income in Q2 2026?"
    assert memory.state.reporting_period == "Q2 2026"
    assert memory.state.business_segment == "U.S. Banking"


def test_memory_discards_old_turns() -> None:
    memory = ConversationMemory(max_turns=2)
    memory.remember("one", "a")
    memory.remember("two", "b")
    memory.remember("three", "c")
    assert [turn.question for turn in memory.turns] == ["two", "three"]
