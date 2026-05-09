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

        # Compute metrics
        retrieved_contexts = result.get("retrieved_contexts", [])
        relevances = [1 if any(label.lower() in str(context).lower() for label in expected_labels) else 0 for context in retrieved_contexts]
        k = cfg.top_k
        ndcg = ndcg_at_k(relevances, k)
        recall = recall_at_k(relevances, k)
        mrr = mrr_at_k(relevances, k)

        ndcg_scores.append(ndcg)
        recall_scores.append(recall)
        mrr_scores.append(mrr)

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

print("\nAverage Metrics:")
print("Avg NDCG@{}: {:.4f}".format(k, np.mean(ndcg_scores)))
print("Avg Recall@{}: {:.4f}".format(k, np.mean(recall_scores)))
print("Avg MRR@{}: {:.4f}".format(k, np.mean(mrr_scores)))

print("\nLeaderboard:")
session.get_leaderboard()