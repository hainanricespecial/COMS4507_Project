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

# ── 6. Run a query ────────────────────────────────────────────────────────────
cfg = RetrievalConfig(mode="hybrid", top_k=5)

with tru_agent as recording:
    result = agent.invoke(
        query="What type is Charizard?",
        query_image_path=None,
        retrieval_config=cfg,
    )

print(result["answer"])
session.get_leaderboard()