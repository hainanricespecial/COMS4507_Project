import json
import numpy as np
from trulens.core import TruSession, Feedback
from trulens.core.otel.instrument import instrument
from trulens.core.feedback.selector import Selector
from trulens.otel.semconv.trace import SpanAttributes
from trulens.apps.custom import TruCustomApp
from trulens.providers.litellm import LiteLLM

from local_agent import PokemonMultimodalAgent, GenerationConfig
from vector_store import MultimodalChromaStore, RetrievalConfig
from pathlib import Path

# ── Metric functions ─────────────────────────────────────────────────────────
def ndcg_at_k(relevances, k):
    if not relevances:
        return 0.0
    dcg = 0.0
    for i in range(min(k, len(relevances))):
        dcg += relevances[i] / np.log2(i + 2)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(k, len(relevances))))
    return dcg / idcg if idcg > 0 else 0.0

def recall_at_k(relevances, k):
    relevant_retrieved = sum(relevances[:k])
    total_relevant = sum(relevances)
    return relevant_retrieved / total_relevant if total_relevant > 0 else 0.0

def mrr_at_k(relevances, k):
    for i in range(min(k, len(relevances))):
        if relevances[i] > 0:
            return 1.0 / (i + 1)
    return 0.0

# ── Helper: extract per-query feedback scores from a TruLens record ──────────
def extract_feedback_scores(session, record):
    """
    Given a TruLens record object, retrieve its feedback results and return
    a dict of { feedback_name: score }.  Falls back to None for any metric
    that hasn't finished computing yet or raised an error.
    """
    scores = {
        "answer_relevance": None,
        "context_relevance": None,
        "groundedness": None,
    }
    try:
        # get_feedback returns a list of FeedbackResult objects for this record
        feedback_results = session.get_feedback(record=record)
        for fb in feedback_results:
            name = fb.feedback_definition.name.lower().replace(" ", "_")
            # fb.result is the aggregated numeric score (0-1)
            if "answer_relevance" in name:
                scores["answer_relevance"] = fb.result
            elif "context_relevance" in name:
                scores["context_relevance"] = fb.result
            elif "groundedness" in name:
                scores["groundedness"] = fb.result
    except Exception as e:
        print(f"  [Warning] Could not extract feedback scores: {e}")
    return scores

# ── 1. TruLens session ──────────────────────────────────────────────────────
session = TruSession()
session.reset_database()

# ── 2. Feedback provider ─────────────────────────────────────────────────────
provider = LiteLLM(
    model_engine="ollama/llava-phi3:3.8b",
    api_base="http://localhost:11434"
)

# ── 3. Feedback functions using OTel Selector API ────────────────────────────
f_answer_relevance = (
    Feedback(provider.relevance, name="Answer Relevance")
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
    Feedback(provider.context_relevance, name="Context Relevance")
    .on_input()
    .on_context(collect_list=False)
    .aggregate(np.mean)
)

f_groundedness = (
    Feedback(provider.groundedness_measure_with_cot_reasons, name="Groundedness")
    .on_context(collect_list=True)
    .on_output()
)

# ── 4. Instrument your agent ─────────────────────────────────────────────────
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

    @instrument()   # ← parentheses required even with no args
    def _answer_node(self, state):
        return super()._answer_node(state)

# ── 5. Wire everything together ───────────────────────────────────────────────
root  = Path(".").resolve()
store = MultimodalChromaStore(root=root)
store.load_index()

agent = InstrumentedAgent(
    store=store,
    generation=GenerationConfig(use_llm=True),
)

tru_agent = TruCustomApp(
    agent,
    app_name="PokemonRAG",
    app_version="v1",
    feedbacks=[f_answer_relevance, f_context_relevance, f_groundedness],
)

# ── 6. Run benchmark queries from JSON ───────────────────────────────────────
cfg = RetrievalConfig(mode="hybrid", top_k=5)
benchmark_path = root / "benchmark_queries.json"

with benchmark_path.open("r", encoding="utf-8") as f:
    benchmark_queries = json.load(f)

ndcg_scores = []
recall_scores = []
mrr_scores = []
answer_relevance_scores = []
context_relevance_scores = []
groundedness_scores = []
results = []

with tru_agent as recording:
    for item in benchmark_queries:
        query = item.get("query")
        query_image = item.get("query_image")
        expected_labels = item.get("expected_labels", [])

        query_image_path = str(root / query_image) if query_image is not None else None

        result = agent.invoke(
            query=query,
            query_image_path=query_image_path,
            retrieval_config=cfg,
        )

        # ── Retrieval metrics (computed directly from retrieved contexts) ──
        retrieved_contexts = result.get("retrieved_contexts", [])
        relevances = [
            1 if any(label.lower() in str(context).lower() for label in expected_labels) else 0
            for context in retrieved_contexts
        ]
        k = cfg.top_k
        ndcg   = ndcg_at_k(relevances, k)
        recall = recall_at_k(relevances, k)
        mrr    = mrr_at_k(relevances, k)

        ndcg_scores.append(ndcg)
        recall_scores.append(recall)
        mrr_scores.append(mrr)

        # ── Per-query LLM-judge scores ─────────────────────────────────────
        # recording.get() returns the most recently completed Record object.
        # We call get_feedback() on it to pull the three scorer results for
        # this individual query before moving on to the next one.
        current_record = recording.get()
        fb_scores = extract_feedback_scores(session, current_record)

        # Accumulate for summary averages (skip None values)
        if fb_scores["answer_relevance"] is not None:
            answer_relevance_scores.append(fb_scores["answer_relevance"])
        if fb_scores["context_relevance"] is not None:
            context_relevance_scores.append(fb_scores["context_relevance"])
        if fb_scores["groundedness"] is not None:
            groundedness_scores.append(fb_scores["groundedness"])

        result_entry = {
            "id": item.get("id"),
            "family": item.get("family"),
            "query": query,
            "query_image": query_image_path,
            "expected_labels": expected_labels,
            "reference_answer": item.get("reference_answer"),
            "model_answer": result.get("answer", str(result)),
            # Retrieval metrics
            "ndcg": ndcg,
            "recall": recall,
            "mrr": mrr,
            # Per-query LLM-judge scores (None if not yet available)
            "answer_relevance": fb_scores["answer_relevance"],
            "context_relevance": fb_scores["context_relevance"],
            "groundedness": fb_scores["groundedness"],
        }
        results.append(result_entry)

        print("\n=== Query ID: {} | Family: {} ===".format(item.get("id"), item.get("family")))
        print("Query:", query)
        if query_image_path:
            print("Image:", query_image_path)
        print("Expected labels:", expected_labels)
        print("Reference answer:", item.get("reference_answer"))
        print("Model answer:", result.get("answer", result))
        print("NDCG@{}: {:.4f}".format(k, ndcg))
        print("Recall@{}: {:.4f}".format(k, recall))
        print("MRR@{}: {:.4f}".format(k, mrr))
        print("Answer Relevance: {}".format(
            "{:.4f}".format(fb_scores["answer_relevance"]) if fb_scores["answer_relevance"] is not None else "N/A"
        ))
        print("Context Relevance: {}".format(
            "{:.4f}".format(fb_scores["context_relevance"]) if fb_scores["context_relevance"] is not None else "N/A"
        ))
        print("Groundedness: {}".format(
            "{:.4f}".format(fb_scores["groundedness"]) if fb_scores["groundedness"] is not None else "N/A"
        ))

print("\nAverage Metrics:")
print("Avg NDCG@{}: {:.4f}".format(k, np.mean(ndcg_scores)))
print("Avg Recall@{}: {:.4f}".format(k, np.mean(recall_scores)))
print("Avg MRR@{}: {:.4f}".format(k, np.mean(mrr_scores)))
print("Avg Answer Relevance: {:.4f}".format(np.mean(answer_relevance_scores)) if answer_relevance_scores else "Avg Answer Relevance: N/A")
print("Avg Context Relevance: {:.4f}".format(np.mean(context_relevance_scores)) if context_relevance_scores else "Avg Context Relevance: N/A")
print("Avg Groundedness: {:.4f}".format(np.mean(groundedness_scores)) if groundedness_scores else "Avg Groundedness: N/A")

print("\nLeaderboard:")
leaderboard_df = session.get_leaderboard()
print(leaderboard_df)

# ── Save results to file ─────────────────────────────────────────────────────
output_file = root / "evaluation_results.json"
with output_file.open("w", encoding="utf-8") as f:
    json.dump({
        "results": results,      # now includes per-query answer/context/groundedness scores
        "summary": {
            "avg_ndcg": float(np.mean(ndcg_scores)),
            "avg_recall": float(np.mean(recall_scores)),
            "avg_mrr": float(np.mean(mrr_scores)),
            "avg_answer_relevance": float(np.mean(answer_relevance_scores)) if answer_relevance_scores else None,
            "avg_context_relevance": float(np.mean(context_relevance_scores)) if context_relevance_scores else None,
            "avg_groundedness": float(np.mean(groundedness_scores)) if groundedness_scores else None,
            "total_queries": len(results),
        },
        "trulens_leaderboard": (
            leaderboard_df.to_dict("records")
            if hasattr(leaderboard_df, "to_dict")
            else str(leaderboard_df)
        ),
    }, f, indent=2)

print("\nResults saved to: {}".format(output_file))