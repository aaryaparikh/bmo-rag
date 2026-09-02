"""Bounded, in-process conversational memory for retrieval query rewriting."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from bmo_rag.generation.openai_responses import (
    OpenAIResponsesClient,
    TelemetryCallback,
)


@dataclass
class ConversationTurn:
    question: str
    answer: str


@dataclass
class ConversationState:
    topic: str = ""
    entity: str = ""
    reporting_period: str = ""
    business_segment: str = ""
    constraints: str = ""
    summary: str = ""


@dataclass
class ConversationMemory:
    max_turns: int = 6
    turns: list[ConversationTurn] = field(default_factory=list)
    state: ConversationState = field(default_factory=ConversationState)

    def remember(self, question: str, answer: str) -> None:
        self.turns.append(ConversationTurn(question=question, answer=answer))
        del self.turns[: max(0, len(self.turns) - self.max_turns)]

    def transcript(self, *, max_chars: int = 8000) -> str:
        text = "\n\n".join(
            f"User: {turn.question}\nAssistant: {turn.answer}" for turn in self.turns
        )
        return text[-max_chars:]

    def prepare_query(
        self,
        question: str,
        client: OpenAIResponsesClient,
        *,
        telemetry: TelemetryCallback | None = None,
    ) -> str:
        """Resolve follow-ups and update compact state without embedding the whole chat."""
        if not self.turns and not any(asdict(self.state).values()):
            # The first question is already standalone and needs no extra API round trip.
            return question
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "standalone_query": {"type": "string"},
                "state": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        key: {"type": "string"} for key in asdict(self.state)
                    },
                    "required": list(asdict(self.state)),
                },
            },
            "required": ["standalone_query", "state"],
        }
        result = client.structured(
            instructions=(
                "Rewrite the newest user message as a concise, standalone retrieval query. "
                "Resolve pronouns and omitted entities, periods, metrics, and business segments "
                "from the conversation. The newest explicit user statement wins. Do not answer "
                "the question or invent details. Update the compact state; use empty strings for "
                "unknown values."
            ),
            input_text=(
                f"Current state:\n{asdict(self.state)}\n\n"
                f"Recent conversation:\n{self.transcript()}\n\n"
                f"Newest user message:\n{question}"
            ),
            schema_name="retrieval_query_context",
            schema=schema,
            telemetry=telemetry,
            operation="query_rewrite",
        )
        state = result.get("state", {})
        self.state = ConversationState(
            **{key: str(state.get(key, "")) for key in asdict(self.state)}
        )
        return str(result.get("standalone_query") or question).strip()
