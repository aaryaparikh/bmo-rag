"""End-to-end conversational RAG pipeline."""

from __future__ import annotations

import sqlite3
import uuid
import warnings
from dataclasses import dataclass

from bmo_rag.generation.memory import ConversationMemory
from bmo_rag.generation.openai_responses import OpenAIResponsesClient, TextDeltaCallback
from bmo_rag.generation.prompts import ANSWER_INSTRUCTIONS, ANSWER_PROMPT
from bmo_rag.indexing.embeddings import resolve_model
from bmo_rag.indexing.qdrant_store import QdrantStore, hybrid_collection_name
from bmo_rag.observability.store import SQLiteObservabilityStore, Trace
from bmo_rag.observability.tokens import count_text_tokens
from bmo_rag.retrieval.context import PackedSource, expand_points, pack_context
from bmo_rag.retrieval.semantic import retrieve_source_aware_hybrid_chunks
from bmo_rag.retrieval.sources import SourceConstraint, resolve_source_constraints


@dataclass(frozen=True)
class ChatAnswer:
    answer: str
    original_question: str
    retrieval_query: str
    sources: tuple[PackedSource, ...]
    requested_sources: tuple[SourceConstraint, ...]
    context_truncated: bool
    trace_id: str | None = None


class RAGChatbot:
    def __init__(
        self,
        *,
        llm: OpenAIResponsesClient,
        memory: ConversationMemory | None = None,
        embedding_model: str = "bge-m3",
        embedding_url: str = "http://127.0.0.1:8000/v1",
        qdrant_url: str = "http://127.0.0.1:6333",
        reranker_url: str = "http://127.0.0.1:8001",
        candidate_k: int = 30,
        seed_k: int = 8,
        max_context_chars: int = 32000,
        max_answer_tokens: int = 5000,
        observability_store: SQLiteObservabilityStore | None = None,
        session_id: str | None = None,
    ) -> None:
        if candidate_k < seed_k:
            raise ValueError("candidate_k must be greater than or equal to seed_k")
        if max_answer_tokens < 1:
            raise ValueError("max_answer_tokens must be greater than zero")
        self.llm = llm
        self.memory = memory or ConversationMemory()
        self.embedding_model = embedding_model
        self.embedding_url = embedding_url
        self.qdrant_url = qdrant_url
        self.reranker_url = reranker_url
        self.candidate_k = candidate_k
        self.seed_k = seed_k
        self.max_context_chars = max_context_chars
        self.max_answer_tokens = max_answer_tokens
        self.observability_store = observability_store
        self.session_id = session_id or str(uuid.uuid4())
        self._source_catalog: dict[str, str | None] | None = None

    def answer(
        self,
        question: str,
        *,
        on_text_delta: TextDeltaCallback | None = None,
    ) -> ChatAnswer:
        trace = self._new_trace(question)
        try:
            return self._answer(question, trace, on_text_delta=on_text_delta)
        except Exception as exc:
            if trace and self.observability_store:
                trace.fail(exc)
                self._save_trace(trace)
            raise

    def _answer(
        self,
        question: str,
        trace: Trace | None,
        *,
        on_text_delta: TextDeltaCallback | None,
    ) -> ChatAnswer:
        if trace:
            with trace.stage("query_normalization"):
                original = question.strip()
        else:
            original = question.strip()
        if not original:
            raise ValueError("question must not be empty")
        if trace:
            with trace.stage("conversation_query_rewrite"):
                retrieval_query = self.memory.prepare_query(
                    original, self.llm, telemetry=trace.record_llm_call
                )
            trace.retrieval_query = retrieval_query
            with trace.stage("model_resolution"):
                spec = resolve_model(self.embedding_model)
        else:
            retrieval_query = self.memory.prepare_query(original, self.llm)
            spec = resolve_model(self.embedding_model)
        collection = hybrid_collection_name(spec.slug, spec.dimension)
        store = QdrantStore(self.qdrant_url)
        if self._source_catalog is None:
            if trace:
                with trace.stage("source_catalog_load"):
                    self._source_catalog = store.source_catalog(collection)
            else:
                self._source_catalog = store.source_catalog(collection)
        elif trace:
            with trace.stage("source_catalog_cache_hit", metadata={"cached": True}):
                pass
        if trace:
            with trace.stage("source_constraint_resolution"):
                source_constraints = resolve_source_constraints(
                    retrieval_query, self._source_catalog
                )
            trace.source_constraints = [
                {"label": item.label, "source_ids": list(item.source_ids)}
                for item in source_constraints
            ]
        else:
            source_constraints = resolve_source_constraints(
                retrieval_query, self._source_catalog
            )
        seeds = retrieve_source_aware_hybrid_chunks(
            retrieval_query,
            source_constraints=source_constraints,
            model=spec.slug,
            top_k=self.seed_k,
            candidate_k=self.candidate_k,
            base_url=self.embedding_url,
            qdrant_url=self.qdrant_url,
            reranker_url=self.reranker_url,
            trace=trace,
        )
        if trace:
            with trace.stage("context_expansion"):
                expanded = expand_points(
                    retrieval_query,
                    seeds,
                    store=store,
                    collection=collection,
                )
            trace.record_chunks("expanded", expanded, lane="context_candidates")
            with trace.stage("context_packing"):
                context = pack_context(expanded, max_chars=self.max_context_chars)
            packed_ids = {source.chunk_id for source in context.sources}
            packed_points = [
                point
                for point in expanded
                if str((point.get("payload") or {}).get("chunk_id") or point.get("id"))
                in packed_ids
            ]
            trace.record_chunks("context", packed_points, lane="llm_evidence")
            trace.evidence_text = context.text
            with trace.stage("evidence_token_count"):
                trace.evidence_tokens, trace.evidence_token_method = count_text_tokens(
                    context.text, self.llm.model
                )
        else:
            expanded = expand_points(
                retrieval_query,
                seeds,
                store=store,
                collection=collection,
            )
            context = pack_context(expanded, max_chars=self.max_context_chars)
        if not context.sources:
            answer = "The available documents do not provide enough evidence to answer that."
        else:
            llm_input = ANSWER_PROMPT.format(
                question=original,
                standalone_query=retrieval_query,
                context=context.text,
            )
            if trace:
                trace.system_prompt = ANSWER_INSTRUCTIONS
                trace.llm_input = llm_input
                with trace.stage(
                    "answer_generation",
                    metadata={"max_output_tokens": self.max_answer_tokens},
                ):
                    answer = self.llm.text(
                        instructions=ANSWER_INSTRUCTIONS,
                        input_text=llm_input,
                        max_output_tokens=self.max_answer_tokens,
                        reasoning_effort="low",
                        telemetry=trace.record_llm_call,
                        operation="answer_generation",
                        on_text_delta=on_text_delta,
                    )
            else:
                answer = self.llm.text(
                    instructions=ANSWER_INSTRUCTIONS,
                    input_text=llm_input,
                    max_output_tokens=self.max_answer_tokens,
                    reasoning_effort="low",
                    on_text_delta=on_text_delta,
                )
        if trace:
            with trace.stage("conversation_memory_update"):
                self.memory.remember(original, answer)
        else:
            self.memory.remember(original, answer)
        result = ChatAnswer(
            answer=answer,
            original_question=original,
            retrieval_query=retrieval_query,
            sources=context.sources,
            requested_sources=tuple(source_constraints),
            context_truncated=context.truncated,
            trace_id=trace.trace_id if trace else None,
        )
        if trace and self.observability_store:
            trace.complete(answer)
            self._save_trace(trace)
        return result

    def _new_trace(self, question: str) -> Trace | None:
        if self.observability_store is None:
            return None
        return self.observability_store.new_trace(
            session_id=self.session_id,
            original_query=question,
            model=self.llm.model,
            embedding_model=self.embedding_model,
            candidate_k=self.candidate_k,
            seed_k=self.seed_k,
        )

    def _save_trace(self, trace: Trace) -> None:
        if self.observability_store is None:
            return
        try:
            self.observability_store.save(trace)
        except sqlite3.Error as store_error:
            warnings.warn(
                f"Could not save RAG trace {trace.trace_id}: {store_error}",
                RuntimeWarning,
                stacklevel=2,
            )
