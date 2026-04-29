from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import re
from typing import Any, Optional, TypedDict
from nltk.corpus import stopwords

try:
    import ollama
except ImportError:
    ollama = None
from langgraph.graph import END, StateGraph

try:
    from .vector_store import MultimodalChromaStore, RetrievalConfig
except ImportError:
    from COMS4507_Project.vector_store import MultimodalChromaStore, RetrievalConfig


class AgentState(TypedDict):
    user_query: str
    query_image_path: Optional[str]
    retrieval_config: dict[str, Any]
    chat_history: list[dict[str, Any]]
    rewritten_query: str
    summarized_query: str
    memory_context: str
    memory_spans: list[dict[str, Any]]
    use_retrieval: bool
    retrieval_decision_reason: str
    retrieved_items: list[dict[str, Any]]
    answer: str
    generation_mode_used: str
    generation_model_used: str
    generation_note: str


@dataclass
class GenerationConfig:
    use_llm: bool = True
    model_name: str = "llava-phi3:3.8b"
    temperature: float = 0.2


class PlantMultimodalAgent:
    # Cache stopwords on initialization to avoid repeated downloads
    _STOPWORDS = None

    @classmethod
    def _get_stopwords(cls) -> set[str]:
        if cls._STOPWORDS is None:
            try:
                cls._STOPWORDS = set(stopwords.words("english"))
            except LookupError:
                # If stopwords corpus is not downloaded, download it
                import nltk
                nltk.download("stopwords", quiet=True)
                cls._STOPWORDS = set(stopwords.words("english"))
        return cls._STOPWORDS

    def __init__(
        self,
        store: MultimodalChromaStore,
        generation: Optional[GenerationConfig] = None,
        enable_query_rewrite: bool = True,
        enable_query_summarize: bool = True,
        max_memory_turns: int = 8,
        max_memory_spans: int = 4,
    ) -> None:
        self.store = store
        self.generation = generation or GenerationConfig()
        self.enable_query_rewrite = enable_query_rewrite
        self.enable_query_summarize = enable_query_summarize
        self.max_memory_turns = max_memory_turns
        self.max_memory_spans = max(1, int(max_memory_spans))
        # Ensure stopwords are loaded
        self._get_stopwords()
        self.app = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("rewrite", self._rewrite_node)
        graph.add_node("decide", self._decide_node)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("answer", self._answer_node)

        graph.set_entry_point("rewrite")
        graph.add_edge("rewrite", "decide")
        graph.add_conditional_edges(
            "decide",
            self._route_after_decision,
            {
                "retrieve": "retrieve",
                "answer": "answer",
            },
        )
        graph.add_edge("retrieve", "answer")
        graph.add_edge("answer", END)
        return graph.compile()

    @staticmethod
    def _compact(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _is_memory_note_request(query: str) -> bool:
        q = (query or "").lower()
        return any(
            phrase in q
            for phrase in [
                "please remember",
                "remember this",
                "remember that",
                "keep this constraint",
                "my constraint",
                "my preference",
                "note that",
            ]
        )

    @staticmethod
    def _is_schedule_request(query: str) -> bool:
        q = (query or "").lower()
        return any(
            phrase in q
            for phrase in [
                "weekly",
                "schedule",
                "once per week",
                "once-weekly",
                "per week",
            ]
        )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", (text or "").lower())

    def _extract_entities(self, text: str) -> list[str]:
        tokens = self._tokenize(text)
        if not tokens:
            return []
        stop = self._get_stopwords()
        counts = Counter(t for t in tokens if t not in stop and len(t) > 2)
        return [w for w, _ in counts.most_common(8)]

    def _summarize_message(self, role: str, text: str) -> str:
        compact = self._compact(text)
        if not compact:
            return ""
        # Keep memory compact and consistent for relevance ranking.
        sentences = re.split(r"(?<=[.!?])\s+", compact)
        first = sentences[0].strip() if sentences else compact
        return f"{role}: {first[:180]}"

    def _select_relevant_memory_spans(self, chat_history: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
        if not chat_history:
            return []

        clipped = chat_history[-(self.max_memory_turns * 2) :]
        query_tokens = set(self._tokenize(query))
        query_entities = set(self._extract_entities(query))

        spans: list[dict[str, Any]] = []
        for i, msg in enumerate(clipped):
            role = str(msg.get("role", "user")).strip().lower()
            content = self._compact(str(msg.get("content", "")))
            if not content:
                continue
            summary = self._summarize_message(role, content)
            entities = self._extract_entities(content)
            span_tokens = set(self._tokenize(summary))
            overlap = len(query_tokens.intersection(span_tokens))
            entity_overlap = len(query_entities.intersection(set(entities)))
            # Recency prior keeps latest turns preferred when relevance ties.
            recency = (i + 1) / max(len(clipped), 1)
            score = (2.0 * entity_overlap) + overlap + (0.3 * recency)
            spans.append(
                {
                    "summary": summary,
                    "entities": entities,
                    "score": score,
                    "raw": content,
                }
            )

        spans.sort(key=lambda x: float(x["score"]), reverse=True)
        return spans[: self.max_memory_spans]

    def _history_to_context(self, chat_history: list[dict[str, Any]]) -> str:
        if not chat_history:
            return ""
        clipped = chat_history[-(self.max_memory_turns * 2) :]
        lines: list[str] = []
        for msg in clipped:
            role = str(msg.get("role", "user")).strip().lower()
            content = self._compact(str(msg.get("content", "")))
            if not content:
                continue
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _heuristic_rewrite_and_summary(self, query: str) -> tuple[str, str]:
        q = self._compact(query)
        lowered = q.lower()
        extra_terms: list[str] = []
        if not any(x in lowered for x in ["symptom", "symptoms"]):
            extra_terms.append("symptoms")
        if not any(x in lowered for x in ["treatment", "cure", "management"]):
            extra_terms.append("treatment")
        if not any(x in lowered for x in ["disease", "blight", "spot", "rust", "mildew"]):
            extra_terms.append("plant disease")

        rewritten = q if not extra_terms else f"{q}. Focus terms: {' '.join(extra_terms)}"

        tokens = re.findall(r"[a-z0-9]+", lowered)
        stop = self._get_stopwords()
        key_tokens = [t for t in tokens if t not in stop]
        summary = " ".join(key_tokens[:14]) if key_tokens else lowered
        return rewritten, summary

    def _llm_available(self) -> bool:
        return bool(self.generation.use_llm and ollama is not None)

    def _rewrite_node(self, state: AgentState) -> AgentState:
        query = state["user_query"]
        history = state.get("chat_history", [])
        memory_spans = self._select_relevant_memory_spans(history, query)
        if memory_spans:
            memory_lines = []
            for idx, span in enumerate(memory_spans, start=1):
                entity_text = ", ".join(span.get("entities", [])[:5]) or "none"
                memory_lines.append(f"M{idx}: {span['summary']} | entities: {entity_text}")
            state["memory_context"] = "\n".join(memory_lines)
        else:
            state["memory_context"] = self._history_to_context(history)
        state["memory_spans"] = memory_spans

        if not (self.enable_query_rewrite or self.enable_query_summarize):
            state["rewritten_query"] = query
            state["summarized_query"] = query
            return state

        if not self._llm_available():
            rewritten, summary = self._heuristic_rewrite_and_summary(query)
            state["rewritten_query"] = rewritten if self.enable_query_rewrite else query
            state["summarized_query"] = summary if self.enable_query_summarize else state["rewritten_query"]
            return state

        prompt = (
            "Rewrite and summarize a user query for sparse retrieval quality.\n"
            "Return exactly two lines in this format:\n"
            "REWRITE: <improved query for retrieval>\n"
            "SUMMARY: <short keyword-rich query>\n\n"
            f"Conversation context:\n{state['memory_context'] or 'N/A'}\n\n"
            f"User query: {query}"
        )

        try:
            response = ollama.chat(
                model=self.generation.model_name,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0},
            )
            content = response["message"]["content"]
            rewrite_line = ""
            summary_line = ""
            for line in content.splitlines():
                if line.upper().startswith("REWRITE:"):
                    rewrite_line = line.split(":", 1)[1].strip()
                if line.upper().startswith("SUMMARY:"):
                    summary_line = line.split(":", 1)[1].strip()

            rewritten = rewrite_line or query
            summarized = summary_line or rewritten
        except Exception:
            rewritten, summarized = self._heuristic_rewrite_and_summary(query)

        state["rewritten_query"] = rewritten if self.enable_query_rewrite else query
        state["summarized_query"] = summarized if self.enable_query_summarize else state["rewritten_query"]
        return state

    def _heuristic_should_retrieve(self, query: str, has_image: bool, has_history: bool) -> tuple[bool, str]:
        if self._is_memory_note_request(query):
            return False, "Preference/constraint follow-up should use memory context first."
        if has_image:
            return True, "User provided a query image, so fresh retrieval is required."
        if not has_history:
            return True, "No conversation memory yet, so retrieval is required."

        q = query.lower().strip()
        retrieval_intent_terms = [
            "disease",
            "symptom",
            "identify",
            "diagnose",
            "cure",
            "treatment",
            "blight",
            "spot",
            "rust",
            "mildew",
            "deficiency",
            "leaf",
            "looks like",
            "look like",
        ]
        followup_terms = [
            "summarize",
            "repeat",
            "rephrase",
            "what did you say",
            "explain again",
            "next step",
            "what about that",
            "and then",
            "why did you",
            "how did you",
            "can you explain",
        ]
        preference_followup_terms = [
            "based on my preference",
            "my preference",
            "using my constraint",
            "my constraint",
            "what should i do first this week",
        ]

        if any(t in q for t in preference_followup_terms):
            return False, "Preference/constraint follow-up should use chat memory context first."

        if any(t in q for t in retrieval_intent_terms):
            return True, "Query contains diagnosis/retrieval intent terms."
        if any(t in q for t in followup_terms):
            return False, "Query looks like a conversational follow-up that can use memory."

        short_followup = len(q.split()) <= 8 and any(x in q for x in ["it", "that", "this", "same"])
        if short_followup:
            return False, "Short referential query likely answerable from chat memory."
        return True, "Defaulted to retrieval for safety."

    def _decide_node(self, state: AgentState) -> AgentState:
        original_query = state["user_query"]
        query = state.get("rewritten_query", original_query)
        has_image = bool(state.get("query_image_path"))
        has_history = bool(state.get("chat_history"))
        heuristic_use_retrieval, heuristic_reason = self._heuristic_should_retrieve(
            original_query,
            has_image,
            has_history,
        )

        if not self._llm_available():
            state["use_retrieval"] = heuristic_use_retrieval
            state["retrieval_decision_reason"] = heuristic_reason
            return state

        gate_prompt = (
            "Decide if a multimodal RAG assistant should perform fresh retrieval or rely on conversation memory.\n"
            "If user asks based on previous preference/constraint, prefer MEMORY unless user asks for fresh evidence.\n"
            "Reply with exactly one line: DECISION: RETRIEVE or DECISION: MEMORY\n"
            "Then one line: REASON: <short reason>.\n\n"
            f"Conversation memory:\n{state.get('memory_context') or 'N/A'}\n\n"
            f"Original user query: {original_query}\n"
            f"Current user query: {query}\n"
            f"User attached image: {'yes' if has_image else 'no'}"
        )

        try:
            response = ollama.chat(
                model=self.generation.model_name,
                messages=[{"role": "user", "content": gate_prompt}],
                options={"temperature": 0.0},
            )
            content = response["message"]["content"]
            decision = "RETRIEVE"
            reason = ""
            for line in content.splitlines():
                up = line.upper().strip()
                if up.startswith("DECISION:"):
                    decision = re.sub(r"[^A-Z]", "", up.split(":", 1)[1].strip())
                if up.startswith("REASON:"):
                    reason = line.split(":", 1)[1].strip()
            use_retrieval = decision != "MEMORY"
        except Exception:
            use_retrieval, reason = heuristic_use_retrieval, heuristic_reason

        reason_lower = reason.lower()
        reason_implies_retrieval = any(
            phrase in reason_lower
            for phrase in [
                "fresh retrieval",
                "requires retrieval",
                "need retrieval",
                "retrieve new",
                "requires fresh",
            ]
        )

        if not use_retrieval and (reason_implies_retrieval or heuristic_use_retrieval):
            use_retrieval = True
            reason = (
                "Overrode memory route for consistency: "
                f"{heuristic_reason if heuristic_use_retrieval else reason}"
            )

        if has_image:
            use_retrieval = True
            reason = reason or "Image queries always force retrieval."

        state["use_retrieval"] = use_retrieval
        state["retrieval_decision_reason"] = reason or (
            "Fresh retrieval selected." if use_retrieval else "Memory-only selected."
        )
        return state

    def _route_after_decision(self, state: AgentState) -> str:
        return "retrieve" if state.get("use_retrieval", True) else "answer"

    def _retrieve_node(self, state: AgentState) -> AgentState:
        if not state.get("use_retrieval", True):
            state["retrieved_items"] = []
            return state

        cfg = RetrievalConfig(**state["retrieval_config"])
        retrieved = self.store.search(
            query_text=state.get("rewritten_query", state["user_query"]),
            query_image_path=state.get("query_image_path"),
            config=cfg,
            sparse_query=state.get("summarized_query", state.get("rewritten_query", state["user_query"])),
        )
        state["retrieved_items"] = retrieved
        return state

    def _make_grounded_prompt(self, state: AgentState) -> str:
        query = state["user_query"]
        rewritten = state.get("rewritten_query", query)
        summarized = state.get("summarized_query", rewritten)
        items = state["retrieved_items"]
        memory_context = state.get("memory_context", "")
        use_retrieval = state.get("use_retrieval", True)
        route_reason = state.get("retrieval_decision_reason", "")

        context_lines: list[str] = []
        for i, item in enumerate(items, start=1):
            text_info = item.get('text', '')[:200] + '...' if item.get('text') else 'No description'
            context_lines.append(
                "\n".join(
                    [
                        f"[R{i}] Result {i}",
                        f"label: {item['label']}",
                        f"description/cure: {text_info}",
                        f"image_name: {item['image_name']}",
                        f"text_score: {item.get('text_score', 0.0):.4f}",
                        f"image_score: {item.get('image_score', 0.0):.4f}",
                        f"dense_score: {item.get('dense_score', 0.0):.4f}",
                        f"sparse_score: {item.get('sparse_score', 0.0):.4f}",
                        f"rrf_score: {item.get('rrf_score', item.get('fusion_score', 0.0)):.4f}",
                    ]
                )
            )

        context = "\n\n".join(context_lines) if context_lines else "No retrieved evidence."

        retrieve_policy = "RETRIEVE" if use_retrieval else "MEMORY_ONLY"

        return (
            "You are a multimodal RAG assistant specializing in plant health diagnosis.\n"
            "Answer using ONLY retrieved evidence and conversation memory. Do not add unsupported details.\n"
            "If evidence is insufficient, say so clearly.\n\n"
            f"User question: {query}\n"
            f"Conversation memory:\n{memory_context or 'No prior conversation.'}\n\n"
            f"Retrieved evidence:\n{context}\n\n"
            "Instructions:\n"
            "1. If the user asks to remember a preference/constraint, acknowledge it explicitly and confirm you will track it.\n"
            "2. If the user asks for a weekly plan, provide a compact schedule based on the evidence and stored constraints.\n"
            "3. For diagnosis requests, use retrieved evidence to identify the issue. Cite sources as [R1], [R2], etc.\n"
            "4. When evidence includes images, describe what the image shows and cite the image result number.\n"
            "5. Provide treatment recommendations ONLY if evidence supports them. Use [R#] citations.\n"
            "6. If evidence is insufficient, respond with: 'Insufficient evidence to diagnose. Please provide more details about [specific aspect].' instead of guessing.\n\n"
            "Response format:\n"
            "- Start by describing what text/image evidence shows (or lack thereof).\n"
            "- If diagnosis is clear: State the issue with [R#] citations, cite treatment/cure steps [R#], suggest next action.\n"
            "- If evidence is mixed/unclear: Explain the uncertainty and ask for clarification.\n"
            "- Keep sentences concise (max 2-3 per section). Do not copy raw retrieval chunks."
        )

    def _verify_output(self, answer: str, state: AgentState) -> tuple[bool, str]:
        text = (answer or "").strip()
        if not text:
            return False, "Empty response."

        lower = text.lower()
        query_l = str(state.get("user_query", "")).lower()
        is_memory_note_request = self._is_memory_note_request(query_l)

        if is_memory_note_request:
            if not any(marker in lower for marker in ["remember", "remembered", "noted", "i’ll remember", "i'll remember", "i will remember"]):
                return False, "Schema check failed: missing memory acknowledgement."
            if not any(marker in lower for marker in ["next step", "next steps", "weekly plan", "action plan"]):
                return False, "Schema check failed: missing next step."
        else:
            schema_markers = ["diagnosis", "next step"]
            missing_schema = [m for m in schema_markers if m not in lower]
            if missing_schema:
                return False, f"Schema check failed: missing {', '.join(missing_schema)}."

        use_retrieval = bool(state.get("use_retrieval", True))
        items = state.get("retrieved_items", [])
        if use_retrieval and items:
            matches = re.findall(r"\[R(\d+)\]", text)
            if not matches:
                return False, "Grounded citation check failed: no [R#] citations found."
            max_idx = len(items)
            for m in matches:
                idx = int(m)
                if idx < 1 or idx > max_idx:
                    return False, f"Grounded citation check failed: [R{idx}] out of range."

        return True, "ok"

    @staticmethod
    def _clean_treatment_text(raw_text: str) -> str:
        text = re.sub(r"\s+", " ", raw_text or "").strip()
        if not text:
            return ""

        treatment_markers = [
            "treatment and cure:",
            "treatment:",
            "recommended treatment/cure:",
            "management:",
        ]
        lowered = text.lower()
        for marker in treatment_markers:
            idx = lowered.find(marker)
            if idx >= 0:
                text = text[idx + len(marker) :].strip()
                break

        if not text:
            return ""

        sentences = re.split(r"(?<=[.!?])\s+", text)
        cleaned: list[str] = []
        for s in sentences:
            s2 = s.strip(" -")
            if not s2:
                continue
            if len(s2) < 25:
                continue
            cleaned.append(s2)
            if len(cleaned) >= 3:
                break

        if cleaned:
            return " ".join(cleaned)
        return text[:260].rstrip(" ,;:-") + "."

    def _heuristic_answer(self, state: AgentState) -> str:
        items = state["retrieved_items"]
        query = state.get("user_query", "")
        query_l = query.lower()
        memory_context = state.get("memory_context", "")

        if self._is_memory_note_request(query):
            if items:
                diagnosis_hint = str(items[0].get("label", "")).strip() or "this plant issue"
            elif memory_context:
                # Fall back to top memory entities when retrieval is intentionally skipped.
                entities: list[str] = []
                for span in state.get("memory_spans", []):
                    entities.extend([str(e) for e in span.get("entities", []) if str(e).strip()])
                uniq_entities: list[str] = []
                for e in entities:
                    if e not in uniq_entities:
                        uniq_entities.append(e)
                diagnosis_hint = " ".join(uniq_entities[:4]).strip() or "this plant issue"
            else:
                diagnosis_hint = "this plant issue"

            memory_ack = f"Yes — I’ll remember your preference and constraint for {diagnosis_hint}."
            if "organic" in query_l or "organic" in memory_context.lower():
                memory_ack = f"Yes — I’ll remember your low-cost organic preference for {diagnosis_hint}."
            elif "once per week" in query_l or "once per week" in memory_context.lower():
                memory_ack = f"Yes — I’ll remember that you can spray only once per week for {diagnosis_hint}."
            if "first" in query_l or "what should i do first" in query_l:
                return (
                    f"**Memory note:** {memory_ack}\n"
                    f"**Likely diagnosis:** {diagnosis_hint}\n"
                    "**First step this week:** remove the most affected leaves, clean tools, and improve airflow first; then consider the lowest-cost organic option that fits your constraint.\n"
                    "**Next step:** Keep monitoring new symptoms before the next treatment window."
                )
            if self._is_schedule_request(query):
                return (
                    f"**Memory note:** {memory_ack}\n"
                    f"**Likely diagnosis:** {diagnosis_hint}\n"
                    "**Weekly plan:**\n"
                    "- Day 1: inspect affected leaves, remove heavily damaged tissue, and keep records.\n"
                    "- Midweek: monitor new lesions, reduce excess moisture, and keep the canopy open.\n"
                    "- Day 7: use your single treatment window if recommended locally, then reassess the crop.\n"
                    "**Next step:** Keep the same once-per-week limit and review symptom changes before the next treatment window."
                )
            return (
                f"**Memory note:** {memory_ack}\n"
                f"**Likely diagnosis:** {diagnosis_hint}\n"
                "**Next step:** I will use your remembered preference and constraint in future advice."
            )

        if not items:
            if not state.get("use_retrieval", True):
                return (
                    "I answered from this chat's memory only. If you want a fresh evidence check against the dataset, "
                    "ask me to re-run retrieval or upload a new image."
                )
            return "I could not retrieve similar examples. Try a clearer plant image and more specific text query."

        top = items[0]
        unique_labels = []
        for item in items:
            if item["label"] not in unique_labels:
                unique_labels.append(item["label"])

        alt = ", ".join(unique_labels[:3])

        # Extract and normalize cure/treatment information from top result text.
        cure_text = self._clean_treatment_text(top.get("text", ""))

        if "concise" in query_l and "treatment" in query_l and "summary" in query_l:
            concise = cure_text or (
                "Prune and remove infected leaves, improve airflow by reducing canopy density, "
                "and apply an approved fungicide or biocontrol option at label rates."
            )
            short = concise[:260].rstrip(" ,;:-") + ("." if not concise.endswith(".") else "")
            return (
                f"**Likely diagnosis:** {top['label']}\n"
                f"**Concise treatment summary:** {short}\n"
                "**Next step:** Remove infected leaves this week and reassess new leaf growth in 5-7 days."
            )

        cure_section = ""
        if cure_text:
            cure_section = f"\n\n**RECOMMENDED TREATMENT/CURE:** {cure_text}"

        evidence = ""
        if items:
            refs = [f"[R{i}]" for i in range(1, min(len(items), 3) + 1)]
            evidence = f"\nEvidence: {', '.join(refs)}"

        return (
            f"**Disease Diagnosis:** {top['label']}\n"
            f"Confidence (RRF score): {top.get('rrf_score', top.get('fusion_score', 0.0)):.3f}\n"
            f"Top match image: {top['image_path']}\n"
            f"Alternative nearby labels: {alt}"
            f"{cure_section}{evidence}\n\n"
            "**Next step:** Inspect the top symptoms against your plant and capture another close-up leaf image for confirmation."
        )

    @staticmethod
    def _looks_noisy_output(text: str) -> bool:
        t = (text or "").strip()
        if not t:
            return True

        tl = t.lower()
        bad_markers = [
            "dense_score",
            "sparse_score",
            "rrf=",
            "the user",
            "the background",
            "the white",
            "0.000000",
        ]
        marker_hits = sum(1 for m in bad_markers if m in tl)
        if marker_hits >= 2:
            return True

        # Detect repetitive low-information fragments.
        chunks = re.findall(r"[a-zA-Z]{3,}", tl)
        if len(chunks) >= 20:
            uniq_ratio = len(set(chunks)) / max(len(chunks), 1)
            if uniq_ratio < 0.35:
                return True

        return False

    def _answer_node(self, state: AgentState) -> AgentState:
        prompt = self._make_grounded_prompt(state)

        if not self.generation.use_llm:
            state["answer"] = self._heuristic_answer(state)
            state["generation_mode_used"] = "heuristic"
            state["generation_model_used"] = "heuristic"
            state["generation_note"] = "Use Ollama VLM is disabled in settings."
            ok, reason = self._verify_output(state["answer"], state)
            if not ok:
                state["answer"] = self._heuristic_answer(state)
                state["generation_note"] += f" Verifier adjusted output: {reason}"
            return state

        if ollama is None:
            state["answer"] = self._heuristic_answer(state)
            state["generation_mode_used"] = "heuristic"
            state["generation_model_used"] = "heuristic"
            state["generation_note"] = "Ollama Python package is not installed."
            ok, reason = self._verify_output(state["answer"], state)
            if not ok:
                state["answer"] = self._heuristic_answer(state)
                state["generation_note"] += f" Verifier adjusted output: {reason}"
            return state

        image_paths: list[str] = []
        # Only attach the user's query image for multimodal generation.
        # Passing retrieved support images often introduces OCR/noise into smaller VLM outputs.
        if state.get("query_image_path") and state.get("use_retrieval", True):
            image_paths = [state["query_image_path"]]

        try:
            response = ollama.chat(
                model=self.generation.model_name,
                messages=[{"role": "user", "content": prompt, "images": image_paths}],
                options={"temperature": self.generation.temperature},
            )
            answer = response["message"]["content"]
            if self._looks_noisy_output(answer):
                state["answer"] = self._heuristic_answer(state)
                state["generation_mode_used"] = "heuristic"
                state["generation_model_used"] = "heuristic"
                state["generation_note"] = "LLM output looked noisy; used cleaned heuristic answer."
                return state

            state["answer"] = answer
            state["generation_mode_used"] = "ollama"
            state["generation_model_used"] = self.generation.model_name
            state["generation_note"] = ""
        except Exception as exc:
            # Some Ollama models are text-only and fail when images are attached.
            # Retry once without images before falling back to heuristic mode.
            try:
                response = ollama.chat(
                    model=self.generation.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": self.generation.temperature},
                )
                answer = response["message"]["content"]
                if self._looks_noisy_output(answer):
                    state["answer"] = self._heuristic_answer(state)
                    state["generation_mode_used"] = "heuristic"
                    state["generation_model_used"] = "heuristic"
                    state["generation_note"] = (
                        "LLM output looked noisy after retry; used cleaned heuristic answer. "
                        f"Image-call error: {exc}"
                    )
                    return state

                state["answer"] = answer
                state["generation_mode_used"] = "ollama"
                state["generation_model_used"] = self.generation.model_name
                state["generation_note"] = (
                    "Ollama image call failed; retried successfully without images. "
                    f"Image-call error: {exc}"
                )
            except Exception as exc2:
                state["answer"] = self._heuristic_answer(state)
                state["generation_mode_used"] = "heuristic"
                state["generation_model_used"] = "heuristic"
                state["generation_note"] = (
                    "Fell back because Ollama call failed with and without images. "
                    f"Image-call error: {exc}; text-call error: {exc2}"
                )

        ok, reason = self._verify_output(state.get("answer", ""), state)
        if not ok:
            state["answer"] = self._heuristic_answer(state)
            prev = state.get("generation_note", "")
            state["generation_note"] = (prev + " " if prev else "") + f"Verifier fallback: {reason}"
            state["generation_mode_used"] = "heuristic"
            state["generation_model_used"] = "heuristic"

        return state

    def invoke(
        self,
        query: str,
        query_image_path: Optional[str],
        retrieval_config: RetrievalConfig,
        chat_history: Optional[list[dict[str, Any]]] = None,
    ) -> AgentState:
        return self.app.invoke(
            {
                "user_query": query,
                "query_image_path": query_image_path,
                "retrieval_config": asdict(retrieval_config),
                "chat_history": chat_history or [],
                "rewritten_query": query,
                "summarized_query": query,
                "memory_context": "",
                "memory_spans": [],
                "use_retrieval": True,
                "retrieval_decision_reason": "",
                "retrieved_items": [],
                "answer": "",
                "generation_mode_used": "",
                "generation_model_used": "",
                "generation_note": "",
            }
        )


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Local LangGraph multimodal plant agent")
    parser.add_argument("--root", default=".")
    parser.add_argument("--query", required=True)
    parser.add_argument("--query-image", default=None)
    parser.add_argument("--mode", default="hybrid", choices=["text_only", "image_only", "hybrid", "auto"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--use-llm", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    store = MultimodalChromaStore(root=root)
    store.load_index()

    agent = PlantMultimodalAgent(
        store=store,
        generation=GenerationConfig(use_llm=args.use_llm),
    )

    cfg = RetrievalConfig(mode=args.mode, top_k=args.top_k)
    out = agent.invoke(args.query, args.query_image, cfg)

    print("\nAnswer:\n")
    print(out["answer"])
    print("\nTop retrieved:\n")
    for i, item in enumerate(out["retrieved_items"], start=1):
        print(i, item["label"], item["fusion_score"], item["image_path"])
