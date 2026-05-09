import json
import numpy as np

# ── Updated imports (non-deprecated) ─────────────────────────────────────────
from trulens.core import TruSession
from trulens.core import Metric                          # replaces Feedback
from trulens.core.otel.instrument import instrument       # OTel-aware: supports span_type & attributes
from trulens.core.feedback.selector import Selector
from trulens.otel.semconv.trace import SpanAttributes
from trulens.apps.app import TruApp                      # replaces TruCustomApp
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
def extract_feedback_scores(record):
    """
    Wait for async feedback computation on this record, then return a dict of
    { feedback_name: score }. Falls back to None for any metric that failed.
    """
    scores = {
        "answer_relevance": None,
        "context_relevance": None,
        "groundedness": None,
    }

    try:
        if hasattr(record, "retrieve_feedback_results"):
            feedback_df = record.retrieve_feedback_results(timeout=60)
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
                name = str(metric_def.name).lower().replace(" ", "_")
                if fb_result is None:
                    continue
                if "answer_relevance" in name:
                    scores["answer_relevance"] = float(fb_result.result)
                elif "context_relevance" in name:
                    scores["context_relevance"] = float(fb_result.result)
                elif "groundedness" in name:
                    scores["groundedness"] = float(fb_result.result)
            return scores

        print("  [Warning] Record object has no feedback retrieval API.")
    except Exception as e:
        print(f"  [Warning] Could not extract feedback scores: {e}")
    return scores

# ── 1. TruLens session ───────────────────────────────────────────────────────
session = TruSession()
session.reset_database()

# ── 2. Feedback provider ─────────────────────────────────────────────────────
provider = LiteLLM(
    model_engine="ollama/llava-phi3:3.8b",
    api_base="http://localhost:11434"
)

# ── 3. Metric functions using OTel Selector API ──────────────────────────────
# Metric replaces the deprecated Feedback class; the chaining API is identical.
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

    @instrument()
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

# TruApp replaces the deprecated TruCustomApp
tru_agent = TruApp(
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

for item in benchmark_queries:
    query           = item.get("query")
    query_image     = item.get("query_image")
    expected_labels = item.get("expected_labels", [])
    query_image_path = str(root / query_image) if query_image is not None else None

    # ── Open a fresh recording context per query ──────────────────────────
    result = None
    try:
        with tru_agent as recording:
            result = agent.invoke(
                query=query,
                query_image_path=query_image_path,
                retrieval_config=cfg,
            )
    except StopIteration:
        # StopIteration can escape TruLens's OTel sync_wrapper when the
        # underlying LiteLLM call fails mid-stream.  The agent result may
        # still be usable if it was set before the exception propagated;
        # otherwise we fall back to an empty dict so the loop can continue.
        print(f"  [Warning] StopIteration caught for query '{query}'. "
              "LiteLLM likely failed mid-call. Skipping LLM-judge scores.")
        if result is None:
            result = {}
    except Exception as e:
        print(f"  [Error] Unexpected error for query '{query}': {e}")
        if result is None:
            result = {}

    # ── Retrieval metrics (computed from retrieved items) ────────────────
    retrieved_items = result.get("retrieved_items", [])
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

    k      = cfg.top_k
    ndcg   = ndcg_at_k(relevances, k)
    recall = recall_at_k(relevances, k)
    mrr    = mrr_at_k(relevances, k)

    ndcg_scores.append(ndcg)
    recall_scores.append(recall)
    mrr_scores.append(mrr)

    # ── Per-query LLM-judge scores ────────────────────────────────────────
    fb_scores = {"answer_relevance": None, "context_relevance": None, "groundedness": None}
    try:
        current_record = recording.get()
        fb_scores = extract_feedback_scores(current_record)
    except Exception as e:
        print(f"  [Warning] Could not retrieve record for feedback: {e}")

    if fb_scores["answer_relevance"] is not None:
        answer_relevance_scores.append(fb_scores["answer_relevance"])
    if fb_scores["context_relevance"] is not None:
        context_relevance_scores.append(fb_scores["context_relevance"])
    if fb_scores["groundedness"] is not None:
        groundedness_scores.append(fb_scores["groundedness"])

    result_entry = {
        "id":               item.get("id"),
        "family":           item.get("family"),
        "query":            query,
        "query_image":      query_image_path,
        "expected_labels":  expected_labels,
        "reference_answer": item.get("reference_answer"),
        "model_answer":     result.get("answer", str(result)),
        # Retrieval metrics
        "ndcg":   ndcg,
        "recall": recall,
        "mrr":    mrr,
        # LLM-judge scores (None if scorer failed / not yet available)
        "answer_relevance":  fb_scores["answer_relevance"],
        "context_relevance": fb_scores["context_relevance"],
        "groundedness":      fb_scores["groundedness"],
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
        "results": results,
        "summary": {
            "avg_ndcg":               float(np.mean(ndcg_scores)),
            "avg_recall":             float(np.mean(recall_scores)),
            "avg_mrr":                float(np.mean(mrr_scores)),
            "avg_answer_relevance":   float(np.mean(answer_relevance_scores))  if answer_relevance_scores  else None,
            "avg_context_relevance":  float(np.mean(context_relevance_scores)) if context_relevance_scores else None,
            "avg_groundedness":       float(np.mean(groundedness_scores))      if groundedness_scores      else None,
            "total_queries":          len(results),
        },
        "trulens_leaderboard": (
            leaderboard_df.to_dict("records")
            if hasattr(leaderboard_df, "to_dict")
            else str(leaderboard_df)
        ),
    }, f, indent=2)

print("\nResults saved to: {}".format(output_file))