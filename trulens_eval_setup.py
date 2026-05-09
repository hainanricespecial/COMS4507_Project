from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue
from typing import Any

import numpy as np
from trulens.apps.app import TruApp
from trulens.core import Metric, TruSession
from trulens.core.feedback.selector import Selector
from trulens.core.otel.instrument import instrument
from trulens.otel.semconv.trace import SpanAttributes
from trulens.providers.litellm import LiteLLM

from local_agent import GenerationConfig, PokemonMultimodalAgent
from vector_store import MultimodalChromaStore, RetrievalConfig


FEEDBACK_KEYS = ("answer_relevance", "context_relevance", "groundedness")


def ndcg_at_k(relevances: list[int], k: int) -> float:
    if not relevances:
        return 0.0
    dcg = sum(relevances[i] / np.log2(i + 2) for i in range(min(k, len(relevances))))
    idcg = sum(1 / np.log2(i + 2) for i in range(min(k, len(relevances))))
    return float(dcg / idcg) if idcg > 0 else 0.0


def recall_at_k(relevances: list[int], k: int) -> float:
    relevant_retrieved = sum(relevances[:k])
    total_relevant = sum(relevances)
    return float(relevant_retrieved / total_relevant) if total_relevant > 0 else 0.0


def mrr_at_k(relevances: list[int], k: int) -> float:
    for i in range(min(k, len(relevances))):
        if relevances[i] > 0:
            return float(1.0 / (i + 1))
    return 0.0


def empty_feedback_scores() -> dict[str, float | None]:
    return {key: None for key in FEEDBACK_KEYS}


def extract_feedback_scores(record: Any) -> dict[str, float | None]:
    """
    Wait for async feedback computation on this record, then return a dict of
    feedback_name -> score. Falls back to None for metrics that failed.
    """
    scores = empty_feedback_scores()

    try:
        if hasattr(record, "retrieve_feedback_results"):
            feedback_timeout = int(os.getenv("TRULENS_FEEDBACK_TIMEOUT", "180"))
            feedback_df = record.retrieve_feedback_results(timeout=feedback_timeout)
            if feedback_df is None or len(feedback_df) == 0:
                print("  [Warning] No feedback results returned for record.")
                return scores

            row = feedback_df.iloc[0] if hasattr(feedback_df, "iloc") else feedback_df[0]
            for key, value in dict(row).items():
                if value is None:
                    continue
                name = str(key).lower().replace(" ", "_")
                if "answer_relevance" in name:
                    scores["answer_relevance"] = float(value)
                elif "context_relevance" in name:
                    scores["context_relevance"] = float(value)
                elif "groundedness" in name:
                    scores["groundedness"] = float(value)
            return scores

        if hasattr(record, "wait_for_feedback_results"):
            fb_map = record.wait_for_feedback_results()
            for metric_def, fb_result in fb_map.items():
                if fb_result is None:
                    continue
                name = str(metric_def.name).lower().replace(" ", "_")
                if "answer_relevance" in name:
                    scores["answer_relevance"] = float(fb_result.result)
                elif "context_relevance" in name:
                    scores["context_relevance"] = float(fb_result.result)
                elif "groundedness" in name:
                    scores["groundedness"] = float(fb_result.result)
            return scores

        print("  [Warning] Record object has no feedback retrieval API.")
    except Exception as exc:
        print(f"  [Warning] Could not extract feedback scores: {exc}")

    return scores


session = TruSession()
session.reset_database()

provider = LiteLLM(
    model_engine="ollama/llava-phi3:3.8b",
    api_base="http://localhost:11434",
)

f_answer_relevance = (
    Metric(provider.relevance, name="Answer Relevance")
    .on({
        "prompt": Selector(
            span_type=SpanAttributes.SpanType.RECORD_ROOT,
            span_attribute=SpanAttributes.RECORD_ROOT.INPUT,
        )
    })
    .on({
        "response": Selector(
            span_type=SpanAttributes.SpanType.RECORD_ROOT,
            span_attribute=SpanAttributes.RECORD_ROOT.OUTPUT,
        )
    })
)

f_context_relevance = (
    Metric(provider.context_relevance, name="Context Relevance")
    .on_input()
    .on_context(collect_list=False)
    .aggregate(np.mean)
)

f_groundedness = (
    Metric(provider.groundedness_measure_with_cot_reasons, name="Groundedness")
    .on_context(collect_list=True)
    .on_output()
)


class InstrumentedAgent(PokemonMultimodalAgent):
    @instrument(
        span_type=SpanAttributes.SpanType.RECORD_ROOT,
        attributes={
            SpanAttributes.RECORD_ROOT.INPUT: "query",
            SpanAttributes.RECORD_ROOT.OUTPUT: "return",
        },
    )
    def invoke(self, query, query_image_path, retrieval_config, chat_history=None):
        return super().invoke(query, query_image_path, retrieval_config, chat_history)

    @instrument(
        span_type=SpanAttributes.SpanType.RETRIEVAL,
        attributes={
            SpanAttributes.RETRIEVAL.QUERY_TEXT: "state",
            SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: "return",
        },
    )
    def _retrieve_node(self, state):
        return super()._retrieve_node(state)

    @instrument()
    def _answer_node(self, state):
        return super()._answer_node(state)


root = Path(".").resolve()


def make_feedbacks(worker_id: int) -> list[Metric]:
    suffix = f"Worker {worker_id}"
    answer_relevance = (
        Metric(provider.relevance, name=f"Answer Relevance {suffix}")
        .on({
            "prompt": Selector(
                span_type=SpanAttributes.SpanType.RECORD_ROOT,
                span_attribute=SpanAttributes.RECORD_ROOT.INPUT,
            )
        })
        .on({
            "response": Selector(
                span_type=SpanAttributes.SpanType.RECORD_ROOT,
                span_attribute=SpanAttributes.RECORD_ROOT.OUTPUT,
            )
        })
    )

    context_relevance = (
        Metric(provider.context_relevance, name=f"Context Relevance {suffix}")
        .on_input()
        .on_context(collect_list=False)
        .aggregate(np.mean)
    )

    groundedness = (
        Metric(provider.groundedness_measure_with_cot_reasons, name=f"Groundedness {suffix}")
        .on_context(collect_list=True)
        .on_output()
    )

    return [answer_relevance, context_relevance, groundedness]


def create_runtime(worker_id: int) -> dict[str, Any]:
    """Load one complete worker runtime before queries are submitted."""
    worker_store = MultimodalChromaStore(root=root)
    worker_store.load_index()
    model_info = worker_store.get_embedding_model_info(ensure_loaded=True)
    if model_info.get("error"):
        raise RuntimeError(f"Could not preload embedding model: {model_info['error']}")

    worker_agent = InstrumentedAgent(
        store=worker_store,
        generation=GenerationConfig(use_llm=True),
    )

    worker_tru_agent = TruApp(
        worker_agent,
        app_name="PokemonRAG",
        app_version=f"v1-worker-{worker_id}",
        feedbacks=make_feedbacks(worker_id),
    )

    return {
        "worker_id": worker_id,
        "agent": worker_agent,
        "tru_agent": worker_tru_agent,
        "model_info": model_info,
    }


def preload_runtimes(max_workers: int) -> Queue:
    runtime_queue: Queue = Queue()
    for worker_id in range(1, max_workers + 1):
        print(f"Preloading worker {worker_id}/{max_workers}: Chroma index + embedding weights...")
        runtime = create_runtime(worker_id)
        model_info = runtime["model_info"]
        active_model = model_info.get("active_model", "unknown")
        model_type = model_info.get("model_type") or "unknown"
        print(f"Worker {worker_id} ready: {active_model} ({model_type})")
        runtime_queue.put(runtime)
    return runtime_queue


def max_workers_from_env() -> int:
    try:
        return max(1, int(os.getenv("TRULENS_MAX_WORKERS", "2")))
    except ValueError:
        return 2


def resolve_query_image(query_image: str | None) -> str | None:
    if query_image is None:
        return None
    return str((root / query_image).resolve())


def retrieved_relevances(
    retrieved_items: list[dict[str, Any]],
    expected_labels: list[str],
) -> list[int]:
    relevances = []
    for item in retrieved_items:
        item_label = str(item.get("label", "")).lower()
        item_text = str(item.get("text", "")).lower()
        item_image = str(item.get("image_path", "")).lower()
        relevant = any(
            label.lower() in item_label
            or label.lower() in item_text
            or label.lower() in item_image
            for label in expected_labels
        )
        relevances.append(1 if relevant else 0)
    return relevances


def evaluate_query_with_runtime(
    index: int,
    case: dict[str, Any],
    cfg: RetrievalConfig,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    query = case.get("query")
    query_image_path = resolve_query_image(case.get("query_image"))
    expected_labels = case.get("expected_labels", [])
    worker_id = runtime["worker_id"]
    agent = runtime["agent"]
    tru_agent = runtime["tru_agent"]

    result: dict[str, Any] | None = None
    recording = None
    error_message = ""

    try:
        with tru_agent as recording:
            result = agent.invoke(
                query=query,
                query_image_path=query_image_path,
                retrieval_config=cfg,
            )
    except StopIteration:
        error_message = (
            "StopIteration caught. LiteLLM likely failed mid-call; "
            "LLM-judge scores may be unavailable."
        )
    except Exception as exc:
        error_message = f"Unexpected error: {exc}"

    result = result or {}
    relevances = retrieved_relevances(result.get("retrieved_items", []), expected_labels)
    k = cfg.top_k
    ndcg = ndcg_at_k(relevances, k)
    recall = recall_at_k(relevances, k)
    mrr = mrr_at_k(relevances, k)

    fb_scores = empty_feedback_scores()
    if recording is not None:
        try:
            fb_scores = extract_feedback_scores(recording.get())
        except Exception as exc:
            error_message = (error_message + " " if error_message else "") + (
                f"Could not retrieve feedback record: {exc}"
            )

    result_entry = {
        "id": case.get("id"),
        "family": case.get("family"),
        "worker_id": worker_id,
        "query": query,
        "query_image": query_image_path,
        "expected_labels": expected_labels,
        "reference_answer": case.get("reference_answer"),
        "model_answer": result.get("answer", str(result)),
        "ndcg": ndcg,
        "recall": recall,
        "mrr": mrr,
        "answer_relevance": fb_scores["answer_relevance"],
        "context_relevance": fb_scores["context_relevance"],
        "groundedness": fb_scores["groundedness"],
    }
    if error_message:
        result_entry["error"] = error_message

    report_lines = [
        "\n=== Query ID: {} | Family: {} ===".format(case.get("id"), case.get("family")),
        f"Worker: {worker_id}",
        f"Query: {query}",
    ]
    if query_image_path:
        report_lines.append(f"Image: {query_image_path}")
    if error_message:
        report_lines.append(f"Warning: {error_message}")
    report_lines.extend([
        f"Expected labels: {expected_labels}",
        f"Reference answer: {case.get('reference_answer')}",
        f"Model answer: {result.get('answer', result)}",
        "NDCG@{}: {:.4f}".format(k, ndcg),
        "Recall@{}: {:.4f}".format(k, recall),
        "MRR@{}: {:.4f}".format(k, mrr),
        "Answer Relevance: {}".format(
            "{:.4f}".format(fb_scores["answer_relevance"])
            if fb_scores["answer_relevance"] is not None
            else "N/A"
        ),
        "Context Relevance: {}".format(
            "{:.4f}".format(fb_scores["context_relevance"])
            if fb_scores["context_relevance"] is not None
            else "N/A"
        ),
        "Groundedness: {}".format(
            "{:.4f}".format(fb_scores["groundedness"])
            if fb_scores["groundedness"] is not None
            else "N/A"
        ),
    ])

    return {
        "index": index,
        "entry": result_entry,
        "feedback": fb_scores,
        "ndcg": ndcg,
        "recall": recall,
        "mrr": mrr,
        "report": "\n".join(report_lines),
    }


def evaluate_query(
    index: int,
    case: dict[str, Any],
    cfg: RetrievalConfig,
    runtime_queue: Queue,
) -> dict[str, Any]:
    runtime = runtime_queue.get()
    try:
        return evaluate_query_with_runtime(index, case, cfg, runtime)
    finally:
        runtime_queue.put(runtime)


def failed_payload(index: int, case: dict[str, Any], exc: Exception) -> dict[str, Any]:
    return {
        "index": index,
        "entry": {
            "id": case.get("id"),
            "family": case.get("family"),
            "query": case.get("query"),
            "query_image": case.get("query_image"),
            "expected_labels": case.get("expected_labels", []),
            "reference_answer": case.get("reference_answer"),
            "model_answer": "",
            "ndcg": 0.0,
            "recall": 0.0,
            "mrr": 0.0,
            "answer_relevance": None,
            "context_relevance": None,
            "groundedness": None,
            "error": f"Worker crashed: {exc}",
        },
        "feedback": empty_feedback_scores(),
        "ndcg": 0.0,
        "recall": 0.0,
        "mrr": 0.0,
        "report": "\n=== Query ID: {} | Family: {} ===\nWorker crashed: {}".format(
            case.get("id"),
            case.get("family"),
            exc,
        ),
    }


def mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def main() -> None:
    cfg = RetrievalConfig(mode="hybrid", top_k=5)
    benchmark_path = root / "benchmark_queries.json"

    with benchmark_path.open("r", encoding="utf-8") as f:
        benchmark_queries = json.load(f)

    max_workers = max_workers_from_env()
    runtime_queue = preload_runtimes(max_workers)
    print(f"Running {len(benchmark_queries)} queries with {max_workers} preloaded worker(s).")

    completed = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(evaluate_query, index, case, cfg, runtime_queue): (index, case)
            for index, case in enumerate(benchmark_queries)
        }

        for future in as_completed(futures):
            index, case = futures[future]
            try:
                payload = future.result()
            except Exception as exc:
                payload = failed_payload(index, case, exc)

            completed.append(payload)
            print(payload["report"])

    completed.sort(key=lambda payload: payload["index"])

    results = []
    ndcg_scores = []
    recall_scores = []
    mrr_scores = []
    answer_relevance_scores = []
    context_relevance_scores = []
    groundedness_scores = []

    for payload in completed:
        entry = payload["entry"]
        fb_scores = payload["feedback"]

        results.append(entry)
        ndcg_scores.append(payload["ndcg"])
        recall_scores.append(payload["recall"])
        mrr_scores.append(payload["mrr"])

        if fb_scores["answer_relevance"] is not None:
            answer_relevance_scores.append(fb_scores["answer_relevance"])
        if fb_scores["context_relevance"] is not None:
            context_relevance_scores.append(fb_scores["context_relevance"])
        if fb_scores["groundedness"] is not None:
            groundedness_scores.append(fb_scores["groundedness"])

    k = cfg.top_k
    print("\nAverage Metrics:")
    print("Avg NDCG@{}: {:.4f}".format(k, float(np.mean(ndcg_scores)) if ndcg_scores else 0.0))
    print("Avg Recall@{}: {:.4f}".format(k, float(np.mean(recall_scores)) if recall_scores else 0.0))
    print("Avg MRR@{}: {:.4f}".format(k, float(np.mean(mrr_scores)) if mrr_scores else 0.0))
    print(
        "Avg Answer Relevance: {:.4f}".format(float(np.mean(answer_relevance_scores)))
        if answer_relevance_scores
        else "Avg Answer Relevance: N/A"
    )
    print(
        "Avg Context Relevance: {:.4f}".format(float(np.mean(context_relevance_scores)))
        if context_relevance_scores
        else "Avg Context Relevance: N/A"
    )
    print(
        "Avg Groundedness: {:.4f}".format(float(np.mean(groundedness_scores)))
        if groundedness_scores
        else "Avg Groundedness: N/A"
    )

    print("\nLeaderboard:")
    leaderboard_df = session.get_leaderboard()
    print(leaderboard_df)

    output_file = root / "evaluation_results.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump({
            "results": results,
            "summary": {
                "avg_ndcg": float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
                "avg_recall": float(np.mean(recall_scores)) if recall_scores else 0.0,
                "avg_mrr": float(np.mean(mrr_scores)) if mrr_scores else 0.0,
                "avg_answer_relevance": mean_or_none(answer_relevance_scores),
                "avg_context_relevance": mean_or_none(context_relevance_scores),
                "avg_groundedness": mean_or_none(groundedness_scores),
                "total_queries": len(results),
                "max_workers": max_workers,
            },
            "trulens_leaderboard": (
                leaderboard_df.to_dict("records")
                if hasattr(leaderboard_df, "to_dict")
                else str(leaderboard_df)
            ),
        }, f, indent=2)

    print("\nResults saved to: {}".format(output_file))


if __name__ == "__main__":
    main()
