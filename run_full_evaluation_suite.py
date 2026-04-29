from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

try:
    import ollama
except ImportError:
    ollama = None

try:
    from .local_agent import GenerationConfig, PlantMultimodalAgent
    from .vector_store import MultimodalChromaStore, RetrievalConfig
except ImportError:
    from local_agent import GenerationConfig, PlantMultimodalAgent
    from vector_store import MultimodalChromaStore, RetrievalConfig


def _parse_csv(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _parse_int_csv(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_weight_pairs(raw: str) -> List[Tuple[float, float]]:
    pairs: List[Tuple[float, float]] = []
    for token in _parse_csv(raw):
        if ":" not in token:
            raise ValueError(f"Invalid weight pair '{token}'. Use format text:image, e.g. 0.5:0.5")
        left, right = token.split(":", 1)
        tw = float(left.strip())
        iw = float(right.strip())
        pairs.append((tw, iw))
    if not pairs:
        raise ValueError("No weight pairs provided.")
    return pairs


def _slug(raw: str) -> str:
    x = re.sub(r"[^a-zA-Z0-9]+", "_", raw.strip().lower())
    x = re.sub(r"_+", "_", x).strip("_")
    return x or "unknown"


def _norm(raw: str) -> str:
    x = raw.lower().replace("_", " ").replace("-", " ")
    x = re.sub(r"[^a-z0-9\s]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def _mean_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    filtered = [v for v in values if v is not None]
    if not filtered:
        return None
    return float(mean(filtered))


def _stable_seed(base_seed: int, *parts: Any) -> int:
    raw = "|".join([str(base_seed)] + [str(p) for p in parts]).encode("utf-8")
    digest = hashlib.sha256(raw).digest()
    return int.from_bytes(digest[:8], "big") % (2**32 - 1)


def _bootstrap_ci(
    values: Iterable[Optional[float]],
    n_samples: int,
    ci_level: float,
    seed: int,
) -> Dict[str, Optional[float]]:
    arr = np.asarray([float(v) for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}

    m = float(arr.mean())
    if arr.size == 1 or n_samples <= 1:
        return {"mean": m, "ci_low": m, "ci_high": m, "n": int(arr.size)}

    alpha = (1.0 - ci_level) / 2.0
    rng = np.random.default_rng(seed)
    sample_idx = rng.integers(0, arr.size, size=(n_samples, arr.size))
    sample_means = arr[sample_idx].mean(axis=1)
    low = float(np.quantile(sample_means, alpha))
    high = float(np.quantile(sample_means, 1.0 - alpha))
    return {"mean": m, "ci_low": low, "ci_high": high, "n": int(arr.size)}


def _ci_payload(stats: Dict[str, Optional[float]], ci_level: float, n_samples: int) -> Dict[str, Optional[float]]:
    return {
        "low": stats.get("ci_low"),
        "high": stats.get("ci_high"),
        "n": stats.get("n"),
        "level": ci_level,
        "bootstrap_samples": n_samples,
    }


def _paired_delta_ci(
    left_values: Iterable[Optional[float]],
    right_values: Iterable[Optional[float]],
    n_samples: int,
    ci_level: float,
    seed: int,
) -> Dict[str, Optional[float]]:
    paired_deltas: List[float] = []
    for left, right in zip(left_values, right_values):
        if left is None or right is None:
            continue
        paired_deltas.append(float(left) - float(right))
    return _bootstrap_ci(paired_deltas, n_samples=n_samples, ci_level=ci_level, seed=seed)


def _record_metric_values(
    summary: Dict[str, Any],
    metric_key: str,
    allowed_families: Optional[set] = None,
) -> List[Optional[float]]:
    values: List[Optional[float]] = []
    for record in summary.get("records", []):
        if allowed_families is not None and record.get("family") not in allowed_families:
            continue
        values.append(record.get(metric_key))
    return values


def _run_key(summary: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        str(summary.get("embedding_model", "")),
        str(summary.get("vector_subdir", "")),
        str(summary.get("generation_model", "")),
    )


def _recall_at_k(retrieved_labels: List[str], expected_labels: List[str]) -> float:
    if not expected_labels:
        return 0.0
    got = {_norm(x) for x in retrieved_labels}
    exp = {_norm(x) for x in expected_labels}
    return 1.0 if got.intersection(exp) else 0.0


def _precision_at_k(retrieved_labels: List[str], expected_labels: List[str]) -> float:
    if not retrieved_labels:
        return 0.0
    exp = {_norm(x) for x in expected_labels}
    if not exp:
        return 0.0
    hits = sum(1 for x in retrieved_labels if _norm(x) in exp)
    return hits / len(retrieved_labels)


def _f1_score(p: float, r: float) -> float:
    if p + r == 0:
        return 0.0
    return 2.0 * p * r / (p + r)


@dataclass
class Experiment:
    name: str
    pipeline: str
    mode: str
    top_k: int
    text_weight: float
    image_weight: float
    enable_query_rewrite: bool = True
    enable_query_summarize: bool = True
    enable_memory: bool = True
    enable_retrieval: bool = True
    use_routing: bool = True


class MetricEngines:
    def __init__(
        self,
        clipscore_model: str,
        disable_bleu: bool,
        disable_rouge: bool,
        disable_bertscore: bool,
        disable_clipscore: bool,
        bertscore_model: str,
    ) -> None:
        self.disable_bleu = disable_bleu
        self.disable_rouge = disable_rouge
        self.disable_bertscore = disable_bertscore
        self.disable_clipscore = disable_clipscore
        self.bertscore_model = bertscore_model

        self._bleu_available = False
        self._rouge_available = False
        self._bertscore_available = False

        self._bleu_smooth = None
        self._rouge = None
        self._bertscore_fn = None

        self._clip_model = None
        self._clip_processor = None
        self._clip_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._init_bleu()
        self._init_rouge()
        self._init_bertscore()
        self._init_clipscore(clipscore_model)

    def _init_bleu(self) -> None:
        if self.disable_bleu:
            return
        try:
            from nltk.translate.bleu_score import SmoothingFunction

            self._bleu_smooth = SmoothingFunction().method1
            self._bleu_available = True
        except Exception:
            self._bleu_available = False

    def _init_rouge(self) -> None:
        if self.disable_rouge:
            return
        try:
            from rouge_score import rouge_scorer

            self._rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
            self._rouge_available = True
        except Exception:
            self._rouge_available = False

    def _init_bertscore(self) -> None:
        if self.disable_bertscore:
            return
        try:
            from bert_score import score as bertscore_score

            self._bertscore_fn = bertscore_score
            self._bertscore_available = True
        except Exception:
            self._bertscore_available = False

    def _init_clipscore(self, clipscore_model: str) -> None:
        if self.disable_clipscore:
            return
        try:
            self._clip_model = CLIPModel.from_pretrained(clipscore_model)
            self._clip_model.to(self._clip_device)
            self._clip_model.eval()
            self._clip_processor = CLIPProcessor.from_pretrained(clipscore_model)
        except Exception:
            self._clip_model = None
            self._clip_processor = None

    @property
    def support(self) -> Dict[str, bool]:
        return {
            "bleu": self._bleu_available,
            "rouge": self._rouge_available,
            "bertscore": self._bertscore_available,
            "clipscore": self._clip_model is not None and self._clip_processor is not None,
        }

    def bleu(self, candidate: str, reference: str) -> Optional[float]:
        if not self._bleu_available:
            return None
        try:
            from nltk.translate.bleu_score import sentence_bleu

            cand_tokens = candidate.split()
            ref_tokens = reference.split()
            if not cand_tokens or not ref_tokens:
                return None
            return float(sentence_bleu([ref_tokens], cand_tokens, smoothing_function=self._bleu_smooth))
        except Exception:
            return None

    def rouge(self, candidate: str, reference: str) -> Tuple[Optional[float], Optional[float]]:
        if not self._rouge_available or self._rouge is None:
            return None, None
        try:
            s = self._rouge.score(reference, candidate)
            return float(s["rouge1"].fmeasure), float(s["rougeL"].fmeasure)
        except Exception:
            return None, None

    def clipscore(self, candidate: str, image_path: Optional[str]) -> Optional[float]:
        if image_path is None or self._clip_model is None or self._clip_processor is None:
            return None
        p = Path(image_path)
        if not p.exists():
            return None
        try:
            with Image.open(p) as img:
                image = img.convert("RGB")
            inputs = self._clip_processor(text=[candidate], images=[image], return_tensors="pt", padding=True)
            inputs = {k: v.to(self._clip_device) for k, v in inputs.items()}
            with torch.no_grad():
                out = self._clip_model(**inputs)
                text_emb = out.text_embeds
                image_emb = out.image_embeds
            score = torch.nn.functional.cosine_similarity(text_emb, image_emb).item()
            return float(score)
        except Exception:
            return None

    def bertscore_batch(self, candidates: List[str], references: List[str]) -> List[Optional[float]]:
        if not self._bertscore_available or self._bertscore_fn is None:
            return [None for _ in candidates]
        if not candidates:
            return []
        try:
            _, _, f1 = self._bertscore_fn(
                candidates,
                references,
                model_type=self.bertscore_model,
                lang="en",
                verbose=False,
                rescale_with_baseline=True,
            )
            return [float(x) for x in f1.cpu().tolist()]
        except Exception:
            return [None for _ in candidates]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full retrieval + generation evaluation suite")
    parser.add_argument("--root", default=".")
    parser.add_argument("--benchmark", default="benchmark_queries_extended.json")
    parser.add_argument(
        "--embedding-models",
        default="openai/clip-vit-base-patch16,openai/clip-vit-base-patch32",
    )
    parser.add_argument(
        "--vector-subdirs",
        default="vector_db/chroma_clip16,vector_db/chroma_clip32",
    )
    parser.add_argument(
        "--generation-models",
        default="llava:7b,llava-phi3:3.8b",
    )
    parser.add_argument("--top-k-values", default="3,5")
    parser.add_argument("--weight-pairs", default="0.7:0.3,0.5:0.5,0.3:0.7")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--memory-turn-window", type=int, default=3)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--save-dir", default="evaluation_results/full_suite")
    parser.add_argument("--parallel-jobs", type=int, default=1)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--ci-level", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=42)

    parser.add_argument("--clipscore-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--bertscore-model", default="distilbert-base-uncased")
    parser.add_argument("--disable-bleu", action="store_true")
    parser.add_argument("--disable-rouge", action="store_true")
    parser.add_argument("--disable-bertscore", action="store_true")
    parser.add_argument("--disable-clipscore", action="store_true")
    parser.add_argument(
        "--disable-toggle-ablations",
        action="store_true",
        help="Disable extra toggle ablations (rewrite/summary/retrieval/memory/routing).",
    )
    return parser.parse_args()


def _resolve_image(root: Path, raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    p = Path(raw)
    p = p if p.is_absolute() else (root / p)
    return str(p.resolve())


def _build_experiments(
    top_ks: List[int],
    weight_pairs: List[Tuple[float, float]],
    include_toggle_ablations: bool,
) -> List[Experiment]:
    exps: List[Experiment] = []

    exps.append(Experiment("generation_only_no_index", "generation_only_no_index", "none", 0, 0.0, 0.0))

    for k in top_ks:
        exps.append(Experiment(f"retrieval_only_text_tk{k}", "retrieval_only", "text_only", k, 1.0, 0.0))
        exps.append(Experiment(f"retrieval_only_image_tk{k}", "retrieval_only", "image_only", k, 0.0, 1.0))

    for k in top_ks:
        for tw, iw in weight_pairs:
            exps.append(Experiment(f"retrieval_only_hybrid_tk{k}_tw{tw}_iw{iw}", "retrieval_only", "hybrid", k, tw, iw))
            exps.append(Experiment(f"rag_hybrid_tk{k}_tw{tw}_iw{iw}", "rag_fixed", "hybrid", k, tw, iw))

    exps.append(Experiment("rag_routed_auto", "rag_routed", "auto", max(top_ks), 0.5, 0.5))
    exps.append(Experiment("rag_memory_auto", "rag_memory", "auto", max(top_ks), 0.5, 0.5))

    if include_toggle_ablations:
        k = max(top_ks)
        exps.extend(
            [
                Experiment(
                    "rag_toggle_full",
                    "rag_toggle",
                    "auto",
                    k,
                    0.5,
                    0.5,
                    enable_query_rewrite=True,
                    enable_query_summarize=True,
                    enable_memory=True,
                    enable_retrieval=True,
                    use_routing=True,
                ),
                Experiment(
                    "rag_toggle_no_rewrite",
                    "rag_toggle",
                    "auto",
                    k,
                    0.5,
                    0.5,
                    enable_query_rewrite=False,
                    enable_query_summarize=True,
                    enable_memory=True,
                    enable_retrieval=True,
                    use_routing=True,
                ),
                Experiment(
                    "rag_toggle_no_summary",
                    "rag_toggle",
                    "auto",
                    k,
                    0.5,
                    0.5,
                    enable_query_rewrite=True,
                    enable_query_summarize=False,
                    enable_memory=True,
                    enable_retrieval=True,
                    use_routing=True,
                ),
                Experiment(
                    "rag_toggle_no_memory",
                    "rag_toggle",
                    "auto",
                    k,
                    0.5,
                    0.5,
                    enable_query_rewrite=True,
                    enable_query_summarize=True,
                    enable_memory=False,
                    enable_retrieval=True,
                    use_routing=True,
                ),
                Experiment(
                    "rag_toggle_no_retrieval",
                    "rag_toggle",
                    "auto",
                    k,
                    0.5,
                    0.5,
                    enable_query_rewrite=True,
                    enable_query_summarize=True,
                    enable_memory=True,
                    enable_retrieval=False,
                    use_routing=True,
                ),
                Experiment(
                    "rag_toggle_no_routing",
                    "rag_toggle",
                    "hybrid",
                    k,
                    0.5,
                    0.5,
                    enable_query_rewrite=True,
                    enable_query_summarize=True,
                    enable_memory=True,
                    enable_retrieval=True,
                    use_routing=False,
                ),
            ]
        )
    return exps


def _route_config(base: RetrievalConfig, case: Dict[str, Any]) -> RetrievalConfig:
    family = str(case.get("family", "")).strip().lower()
    has_image = bool(case.get("query_image"))

    if family == "factual_retrieval" and not has_image:
        return RetrievalConfig(
            mode="text_only",
            top_k=base.top_k,
            text_weight=1.0,
            image_weight=0.0,
            text_threshold=base.text_threshold,
            image_threshold=base.image_threshold,
        )

    if family == "cross_modal_retrieval" or has_image:
        return RetrievalConfig(
            mode="hybrid",
            top_k=max(base.top_k, 5),
            text_weight=0.4,
            image_weight=0.6,
            text_threshold=base.text_threshold,
            image_threshold=base.image_threshold,
        )

    if family in {"analytical_multi_hop", "conversational_follow_up", "memory_multi_turn"}:
        return RetrievalConfig(
            mode="hybrid",
            top_k=max(base.top_k, 8),
            text_weight=0.6,
            image_weight=0.4,
            text_threshold=base.text_threshold,
            image_threshold=base.image_threshold,
        )

    return base


def _invoke_generation_only(
    query: str,
    query_image: Optional[str],
    generation_model: str,
    temperature: float,
) -> Dict[str, Any]:
    if ollama is None:
        return {
            "answer": "",
            "generation_mode_used": "failed",
            "generation_model_used": generation_model,
            "generation_note": "Ollama Python package is not installed.",
        }

    message: Dict[str, Any] = {"role": "user", "content": query}
    if query_image:
        message["images"] = [query_image]

    try:
        resp = ollama.chat(model=generation_model, messages=[message], options={"temperature": temperature})
        return {
            "answer": resp["message"]["content"],
            "generation_mode_used": "ollama",
            "generation_model_used": generation_model,
            "generation_note": "",
        }
    except Exception as exc:
        try:
            resp = ollama.chat(
                model=generation_model,
                messages=[{"role": "user", "content": query}],
                options={"temperature": temperature},
            )
            return {
                "answer": resp["message"]["content"],
                "generation_mode_used": "ollama",
                "generation_model_used": generation_model,
                "generation_note": f"Image call failed; retried text-only. Error: {exc}",
            }
        except Exception as exc2:
            return {
                "answer": "",
                "generation_mode_used": "failed",
                "generation_model_used": generation_model,
                "generation_note": f"Generation failed with and without image. image_error={exc}; text_error={exc2}",
            }


def _build_memory_query(
    query: str,
    thread_id: Optional[str],
    memory_state: Dict[str, List[str]],
    turn_window: int,
) -> str:
    if not thread_id:
        return query
    history = memory_state.get(thread_id, [])
    if not history:
        return query
    window = history[-turn_window:]
    memory_block = "\n".join(window)
    return (
        "Conversation memory context:\n"
        f"{memory_block}\n\n"
        "Current user query:\n"
        f"{query}"
    )


def _clipscore_image_for_record(root: Path, query_image: Optional[str], retrieved_items: List[Dict[str, Any]]) -> Optional[str]:
    if query_image:
        return query_image
    if retrieved_items:
        p = Path(retrieved_items[0].get("image_path", ""))
        if not p.is_absolute():
            p = (root / p).resolve()
        return str(p)
    return None


def _summarize_records(
    records: List[Dict[str, Any]],
    embedding_model: str,
    vector_subdir: str,
    generation_model: str,
    experiment: Experiment,
    bootstrap_samples: int,
    ci_level: float,
    bootstrap_seed: int,
) -> Dict[str, Any]:
    family_buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        family_buckets[r.get("family", "unknown")].append(r)

    def _metric(
        values: Iterable[Optional[float]],
        metric_name: str,
        family_name: Optional[str] = None,
    ) -> Tuple[Optional[float], Dict[str, Optional[float]]]:
        seed = _stable_seed(
            bootstrap_seed,
            embedding_model,
            vector_subdir,
            generation_model,
            experiment.name,
            family_name or "overall",
            metric_name,
        )
        stats = _bootstrap_ci(
            values=values,
            n_samples=bootstrap_samples,
            ci_level=ci_level,
            seed=seed,
        )
        return stats.get("mean"), _ci_payload(stats, ci_level=ci_level, n_samples=bootstrap_samples)

    family_summary: Dict[str, Any] = {}
    for fam, rs in family_buckets.items():
        fam_recall, fam_recall_ci = _metric([r.get("recall_at_k") for r in rs], "avg_recall_at_k", fam)
        fam_precision, fam_precision_ci = _metric([r.get("precision_at_k") for r in rs], "avg_precision_at_k", fam)
        fam_f1, fam_f1_ci = _metric([r.get("f1_retrieval") for r in rs], "avg_f1_retrieval", fam)
        fam_bleu, fam_bleu_ci = _metric([r.get("bleu") for r in rs], "avg_bleu", fam)
        fam_rouge_l, fam_rouge_l_ci = _metric([r.get("rouge_l") for r in rs], "avg_rouge_l", fam)
        fam_bert, fam_bert_ci = _metric([r.get("bertscore_f1") for r in rs], "avg_bertscore_f1", fam)
        fam_clip, fam_clip_ci = _metric([r.get("clipscore") for r in rs], "avg_clipscore", fam)
        fam_latency, fam_latency_ci = _metric([r.get("latency_ms") for r in rs], "avg_latency_ms", fam)
        fam_ollama, fam_ollama_ci = _metric(
            [1.0 if r.get("generation_mode_used") == "ollama" else 0.0 for r in rs],
            "ollama_usage_rate",
            fam,
        )
        family_summary[fam] = {
            "n": len(rs),
            "avg_recall_at_k": fam_recall,
            "avg_recall_at_k_ci": fam_recall_ci,
            "avg_precision_at_k": fam_precision,
            "avg_precision_at_k_ci": fam_precision_ci,
            "avg_f1_retrieval": fam_f1,
            "avg_f1_retrieval_ci": fam_f1_ci,
            "avg_bleu": fam_bleu,
            "avg_bleu_ci": fam_bleu_ci,
            "avg_rouge_l": fam_rouge_l,
            "avg_rouge_l_ci": fam_rouge_l_ci,
            "avg_bertscore_f1": fam_bert,
            "avg_bertscore_f1_ci": fam_bert_ci,
            "avg_clipscore": fam_clip,
            "avg_clipscore_ci": fam_clip_ci,
            "avg_latency_ms": fam_latency,
            "avg_latency_ms_ci": fam_latency_ci,
            "ollama_usage_rate": fam_ollama,
            "ollama_usage_rate_ci": fam_ollama_ci,
        }

    overall_recall, overall_recall_ci = _metric([r.get("recall_at_k") for r in records], "avg_recall_at_k")
    overall_precision, overall_precision_ci = _metric([r.get("precision_at_k") for r in records], "avg_precision_at_k")
    overall_f1, overall_f1_ci = _metric([r.get("f1_retrieval") for r in records], "avg_f1_retrieval")
    overall_bleu, overall_bleu_ci = _metric([r.get("bleu") for r in records], "avg_bleu")
    overall_rouge_l, overall_rouge_l_ci = _metric([r.get("rouge_l") for r in records], "avg_rouge_l")
    overall_bert, overall_bert_ci = _metric([r.get("bertscore_f1") for r in records], "avg_bertscore_f1")
    overall_clip, overall_clip_ci = _metric([r.get("clipscore") for r in records], "avg_clipscore")
    overall_latency, overall_latency_ci = _metric([r.get("latency_ms") for r in records], "avg_latency_ms")
    overall_ollama, overall_ollama_ci = _metric(
        [1.0 if r.get("generation_mode_used") == "ollama" else 0.0 for r in records],
        "ollama_usage_rate",
    )
    overall_fallback, overall_fallback_ci = _metric(
        [1.0 if r.get("generation_mode_used") != "ollama" else 0.0 for r in records],
        "fallback_rate",
    )

    summary = {
        "embedding_model": embedding_model,
        "vector_subdir": vector_subdir,
        "generation_model": generation_model,
        "experiment": {
            "name": experiment.name,
            "pipeline": experiment.pipeline,
            "mode": experiment.mode,
            "top_k": experiment.top_k,
            "text_weight": experiment.text_weight,
            "image_weight": experiment.image_weight,
            "enable_query_rewrite": experiment.enable_query_rewrite,
            "enable_query_summarize": experiment.enable_query_summarize,
            "enable_memory": experiment.enable_memory,
            "enable_retrieval": experiment.enable_retrieval,
            "use_routing": experiment.use_routing,
        },
        "n_cases": len(records),
        "ci_config": {
            "level": ci_level,
            "bootstrap_samples": bootstrap_samples,
        },
        "avg_recall_at_k": overall_recall,
        "avg_recall_at_k_ci": overall_recall_ci,
        "avg_precision_at_k": overall_precision,
        "avg_precision_at_k_ci": overall_precision_ci,
        "avg_f1_retrieval": overall_f1,
        "avg_f1_retrieval_ci": overall_f1_ci,
        "avg_bleu": overall_bleu,
        "avg_bleu_ci": overall_bleu_ci,
        "avg_rouge_l": overall_rouge_l,
        "avg_rouge_l_ci": overall_rouge_l_ci,
        "avg_bertscore_f1": overall_bert,
        "avg_bertscore_f1_ci": overall_bert_ci,
        "avg_clipscore": overall_clip,
        "avg_clipscore_ci": overall_clip_ci,
        "avg_latency_ms": overall_latency,
        "avg_latency_ms_ci": overall_latency_ci,
        "ollama_usage_rate": overall_ollama,
        "ollama_usage_rate_ci": overall_ollama_ci,
        "fallback_rate": overall_fallback,
        "fallback_rate_ci": overall_fallback_ci,
        "family_summary": family_summary,
        "records": records,
    }
    return summary


def _save_summary(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _build_question_answers(
    all_summaries: List[Dict[str, Any]],
    bootstrap_samples: int,
    ci_level: float,
    bootstrap_seed: int,
) -> Dict[str, Any]:
    def _pick(predicate) -> List[Dict[str, Any]]:
        return [s for s in all_summaries if predicate(s)]

    def _best_by(metric: str, candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        usable = [c for c in candidates if c.get(metric) is not None]
        if not usable:
            return None
        return max(usable, key=lambda x: x.get(metric, float("-inf")))

    def _run_descriptor(summary: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if summary is None:
            return None
        exp = summary.get("experiment", {})
        return {
            "embedding_model": summary.get("embedding_model"),
            "vector_subdir": summary.get("vector_subdir"),
            "generation_model": summary.get("generation_model"),
            "experiment_name": exp.get("name"),
            "pipeline": exp.get("pipeline"),
            "mode": exp.get("mode"),
            "top_k": exp.get("top_k"),
            "text_weight": exp.get("text_weight"),
            "image_weight": exp.get("image_weight"),
            "enable_query_rewrite": exp.get("enable_query_rewrite"),
            "enable_query_summarize": exp.get("enable_query_summarize"),
            "enable_memory": exp.get("enable_memory"),
            "enable_retrieval": exp.get("enable_retrieval"),
            "use_routing": exp.get("use_routing"),
        }

    def _delta_conclusion(
        stats: Dict[str, Optional[float]],
        positive: str,
        negative: str,
        inconclusive: str,
    ) -> str:
        low = stats.get("ci_low")
        high = stats.get("ci_high")
        if low is None or high is None:
            return "Insufficient data"
        if low > 0:
            return positive
        if high < 0:
            return negative
        return inconclusive

    def _complex_family_avg(summary: Dict[str, Any]) -> Optional[float]:
        vals: List[float] = []
        for fam in {"analytical_multi_hop", "conversational_follow_up"}:
            payload = summary.get("family_summary", {}).get(fam, {})
            v = payload.get("avg_bertscore_f1")
            if v is not None:
                vals.append(float(v))
        return float(mean(vals)) if vals else None

    def _family_avg(summary: Dict[str, Any], family: str, metric: str) -> Optional[float]:
        payload = summary.get("family_summary", {}).get(family, {})
        v = payload.get(metric)
        return None if v is None else float(v)

    def _map_by_key(
        runs: List[Dict[str, Any]],
        value_fn,
    ) -> Dict[Tuple[str, str, str], float]:
        out: Dict[Tuple[str, str, str], float] = {}
        for run in runs:
            v = value_fn(run)
            if v is not None:
                out[_run_key(run)] = float(v)
        return out

    def _paired_map_delta(
        left_map: Dict[Tuple[str, str, str], float],
        right_map: Dict[Tuple[str, str, str], float],
        tag: str,
    ) -> Dict[str, Optional[float]]:
        keys = sorted(set(left_map.keys()).intersection(right_map.keys()))
        deltas = [left_map[k] - right_map[k] for k in keys]
        return _bootstrap_ci(
            values=deltas,
            n_samples=bootstrap_samples,
            ci_level=ci_level,
            seed=_stable_seed(bootstrap_seed, "question_delta", tag),
        )

    text_runs = _pick(lambda s: s["experiment"]["pipeline"] == "retrieval_only" and s["experiment"]["mode"] == "text_only")
    image_runs = _pick(lambda s: s["experiment"]["pipeline"] == "retrieval_only" and s["experiment"]["mode"] == "image_only")
    hybrid_runs = _pick(lambda s: s["experiment"]["pipeline"] in {"retrieval_only", "rag_fixed"} and s["experiment"]["mode"] == "hybrid")
    routed_runs = _pick(lambda s: s["experiment"]["pipeline"] == "rag_routed")
    memory_runs = _pick(lambda s: s["experiment"]["pipeline"] == "rag_memory")
    gen_only_runs = _pick(lambda s: s["experiment"]["pipeline"] == "generation_only_no_index")

    best_text = _best_by("avg_recall_at_k", text_runs)
    best_image = _best_by("avg_recall_at_k", image_runs)
    best_hybrid = _best_by("avg_recall_at_k", hybrid_runs)

    fixed_hybrid_baseline_runs = _pick(
        lambda s: s["experiment"]["pipeline"] == "rag_fixed"
        and s["experiment"]["mode"] == "hybrid"
        and s["experiment"]["top_k"] == 5
        and abs(float(s["experiment"]["text_weight"]) - 0.5) < 1e-9
        and abs(float(s["experiment"]["image_weight"]) - 0.5) < 1e-9
    )

    best_single = None
    single_candidates = [x for x in [best_text, best_image] if x is not None and x.get("avg_recall_at_k") is not None]
    if single_candidates:
        best_single = max(single_candidates, key=lambda x: x.get("avg_recall_at_k", float("-inf")))

    hybrid_vs_single_delta = {
        "mean": None,
        "ci_low": None,
        "ci_high": None,
        "n": 0,
    }
    if best_hybrid is not None and best_single is not None:
        hybrid_vs_single_delta = _paired_delta_ci(
            left_values=_record_metric_values(best_hybrid, "recall_at_k"),
            right_values=_record_metric_values(best_single, "recall_at_k"),
            n_samples=bootstrap_samples,
            ci_level=ci_level,
            seed=_stable_seed(bootstrap_seed, "hybrid_vs_single_recall"),
        )

    routed_complex_map = _map_by_key(routed_runs, _complex_family_avg)
    fixed_complex_map = _map_by_key(fixed_hybrid_baseline_runs, _complex_family_avg)
    routed_complex_stats = _bootstrap_ci(
        values=routed_complex_map.values(),
        n_samples=bootstrap_samples,
        ci_level=ci_level,
        seed=_stable_seed(bootstrap_seed, "routed_complex"),
    )
    fixed_complex_stats = _bootstrap_ci(
        values=fixed_complex_map.values(),
        n_samples=bootstrap_samples,
        ci_level=ci_level,
        seed=_stable_seed(bootstrap_seed, "fixed_complex"),
    )
    routed_vs_fixed_complex_delta = _paired_map_delta(routed_complex_map, fixed_complex_map, "routed_vs_fixed_complex")

    memory_map = _map_by_key(memory_runs, lambda s: _family_avg(s, "memory_multi_turn", "avg_bertscore_f1"))
    routed_memory_map = _map_by_key(routed_runs, lambda s: _family_avg(s, "memory_multi_turn", "avg_bertscore_f1"))
    memory_stats = _bootstrap_ci(
        values=memory_map.values(),
        n_samples=bootstrap_samples,
        ci_level=ci_level,
        seed=_stable_seed(bootstrap_seed, "memory_pipeline"),
    )
    routed_memory_stats = _bootstrap_ci(
        values=routed_memory_map.values(),
        n_samples=bootstrap_samples,
        ci_level=ci_level,
        seed=_stable_seed(bootstrap_seed, "routed_memory_baseline"),
    )
    memory_vs_routed_delta = _paired_map_delta(memory_map, routed_memory_map, "memory_vs_routed")

    gen_map = _map_by_key(gen_only_runs, lambda s: s.get("avg_bertscore_f1"))
    fixed_map = _map_by_key(fixed_hybrid_baseline_runs, lambda s: s.get("avg_bertscore_f1"))
    routed_map = _map_by_key(routed_runs, lambda s: s.get("avg_bertscore_f1"))

    gen_stats = _bootstrap_ci(
        values=gen_map.values(),
        n_samples=bootstrap_samples,
        ci_level=ci_level,
        seed=_stable_seed(bootstrap_seed, "decompose_gen_only"),
    )
    fixed_stats = _bootstrap_ci(
        values=fixed_map.values(),
        n_samples=bootstrap_samples,
        ci_level=ci_level,
        seed=_stable_seed(bootstrap_seed, "decompose_retrieve_answer"),
    )
    routed_stats = _bootstrap_ci(
        values=routed_map.values(),
        n_samples=bootstrap_samples,
        ci_level=ci_level,
        seed=_stable_seed(bootstrap_seed, "decompose_plan_retrieve_answer"),
    )

    retrieve_vs_gen_delta = _paired_map_delta(fixed_map, gen_map, "retrieve_vs_gen")
    plan_vs_retrieve_delta = _paired_map_delta(routed_map, fixed_map, "plan_vs_retrieve")
    plan_vs_gen_delta = _paired_map_delta(routed_map, gen_map, "plan_vs_gen")

    text_ci = None if best_text is None else best_text.get("avg_recall_at_k_ci")
    text_conclusion = "Insufficient data"
    if best_text is not None and text_ci is not None:
        low = text_ci.get("low")
        high = text_ci.get("high")
        if high is not None and high < 0.6:
            text_conclusion = "Likely insufficient (CI below 0.6 threshold)"
        elif low is not None and low >= 0.6:
            text_conclusion = "Potentially sufficient (CI above 0.6 threshold)"
        else:
            text_conclusion = "Uncertain around the 0.6 threshold"

    image_ci = None if best_image is None else best_image.get("avg_recall_at_k_ci")
    image_conclusion = "Insufficient data"
    if best_image is not None and image_ci is not None:
        low = image_ci.get("low")
        high = image_ci.get("high")
        if low is not None and low > 0:
            image_conclusion = "Yes, image-only supports retrieval"
        elif high is not None and high <= 0:
            image_conclusion = "Not supported by current results"
        else:
            image_conclusion = "Likely yes but uncertain"

    answers: Dict[str, Any] = {
        "is_text_only_indexing_sufficient_for_multimodal_qa": {
            "best_text_recall": None if best_text is None else best_text.get("avg_recall_at_k"),
            "best_text_recall_ci": text_ci,
            "best_text_run": _run_descriptor(best_text),
            "conclusion": text_conclusion,
        },
        "can_image_only_embeddings_support_retrieval": {
            "best_image_recall": None if best_image is None else best_image.get("avg_recall_at_k"),
            "best_image_recall_ci": image_ci,
            "best_image_run": _run_descriptor(best_image),
            "conclusion": image_conclusion,
        },
        "does_hybrid_outperform_single_space": {
            "best_hybrid_recall": None if best_hybrid is None else best_hybrid.get("avg_recall_at_k"),
            "best_hybrid_recall_ci": None if best_hybrid is None else best_hybrid.get("avg_recall_at_k_ci"),
            "best_hybrid_run": _run_descriptor(best_hybrid),
            "best_single_space_recall": max(
                [x for x in [
                    None if best_text is None else best_text.get("avg_recall_at_k"),
                    None if best_image is None else best_image.get("avg_recall_at_k"),
                ] if x is not None],
                default=None,
            ),
            "best_single_space_run": _run_descriptor(best_single),
            "recall_delta_hybrid_minus_best_single": hybrid_vs_single_delta.get("mean"),
            "recall_delta_hybrid_minus_best_single_ci": _ci_payload(
                hybrid_vs_single_delta,
                ci_level=ci_level,
                n_samples=bootstrap_samples,
            ),
        },
        "does_agentic_routing_improve_complex_queries": {
            "routed_complex_bertscore": routed_complex_stats.get("mean"),
            "routed_complex_bertscore_ci": _ci_payload(routed_complex_stats, ci_level=ci_level, n_samples=bootstrap_samples),
            "fixed_complex_bertscore": fixed_complex_stats.get("mean"),
            "fixed_complex_bertscore_ci": _ci_payload(fixed_complex_stats, ci_level=ci_level, n_samples=bootstrap_samples),
            "delta_routed_minus_fixed": routed_vs_fixed_complex_delta.get("mean"),
            "delta_routed_minus_fixed_ci": _ci_payload(
                routed_vs_fixed_complex_delta,
                ci_level=ci_level,
                n_samples=bootstrap_samples,
            ),
            "conclusion": _delta_conclusion(
                routed_vs_fixed_complex_delta,
                positive="Likely yes (routing improves complex-query BERTScore)",
                negative="Likely no (routing is worse on complex-query BERTScore)",
                inconclusive="Inconclusive (CI overlaps zero)",
            ),
        },
        "does_memory_help_multi_turn_personalised_interactions": {
            "memory_pipeline_bertscore": memory_stats.get("mean"),
            "memory_pipeline_bertscore_ci": _ci_payload(memory_stats, ci_level=ci_level, n_samples=bootstrap_samples),
            "routed_without_memory_bertscore": routed_memory_stats.get("mean"),
            "routed_without_memory_bertscore_ci": _ci_payload(
                routed_memory_stats,
                ci_level=ci_level,
                n_samples=bootstrap_samples,
            ),
            "delta_memory_minus_routed": memory_vs_routed_delta.get("mean"),
            "delta_memory_minus_routed_ci": _ci_payload(
                memory_vs_routed_delta,
                ci_level=ci_level,
                n_samples=bootstrap_samples,
            ),
            "conclusion": _delta_conclusion(
                memory_vs_routed_delta,
                positive="Likely yes (memory helps multi-turn personalization)",
                negative="Likely no (memory hurts multi-turn personalization)",
                inconclusive="Inconclusive (CI overlaps zero)",
            ),
        },
        "what_is_gained_by_separating_retrieval_planning_answering": {
            "generation_only_bertscore": gen_stats.get("mean"),
            "generation_only_bertscore_ci": _ci_payload(gen_stats, ci_level=ci_level, n_samples=bootstrap_samples),
            "retrieve_answer_bertscore": fixed_stats.get("mean"),
            "retrieve_answer_bertscore_ci": _ci_payload(fixed_stats, ci_level=ci_level, n_samples=bootstrap_samples),
            "plan_retrieve_answer_bertscore": routed_stats.get("mean"),
            "plan_retrieve_answer_bertscore_ci": _ci_payload(routed_stats, ci_level=ci_level, n_samples=bootstrap_samples),
            "delta_retrieve_minus_generation": retrieve_vs_gen_delta.get("mean"),
            "delta_retrieve_minus_generation_ci": _ci_payload(
                retrieve_vs_gen_delta,
                ci_level=ci_level,
                n_samples=bootstrap_samples,
            ),
            "delta_plan_minus_retrieve": plan_vs_retrieve_delta.get("mean"),
            "delta_plan_minus_retrieve_ci": _ci_payload(
                plan_vs_retrieve_delta,
                ci_level=ci_level,
                n_samples=bootstrap_samples,
            ),
            "delta_plan_minus_generation": plan_vs_gen_delta.get("mean"),
            "delta_plan_minus_generation_ci": _ci_payload(
                plan_vs_gen_delta,
                ci_level=ci_level,
                n_samples=bootstrap_samples,
            ),
        },
    }

    answers["does_hybrid_outperform_single_space"]["conclusion"] = _delta_conclusion(
        hybrid_vs_single_delta,
        positive="Likely yes (hybrid outperforms best single space)",
        negative="Likely no (hybrid underperforms best single space)",
        inconclusive="Inconclusive (CI overlaps zero)",
    )

    answers["what_is_gained_by_separating_retrieval_planning_answering"]["conclusion"] = {
        "retrieve_vs_generation": _delta_conclusion(
            retrieve_vs_gen_delta,
            positive="Retrieve+answer improves over generation-only",
            negative="Retrieve+answer is worse than generation-only",
            inconclusive="Inconclusive",
        ),
        "plan_vs_retrieve": _delta_conclusion(
            plan_vs_retrieve_delta,
            positive="Planning helps over retrieve+answer",
            negative="Planning hurts vs retrieve+answer",
            inconclusive="Inconclusive",
        ),
        "plan_vs_generation": _delta_conclusion(
            plan_vs_gen_delta,
            positive="Planning+retrieve+answer improves over generation-only",
            negative="Planning+retrieve+answer is worse than generation-only",
            inconclusive="Inconclusive",
        ),
    }

    return answers


def _write_csv(path: Path, summaries: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "embedding_model",
        "vector_subdir",
        "generation_model",
        "experiment_name",
        "pipeline",
        "mode",
        "top_k",
        "text_weight",
        "image_weight",
        "n_cases",
        "avg_recall_at_k",
        "avg_precision_at_k",
        "avg_f1_retrieval",
        "avg_bleu",
        "avg_rouge_l",
        "avg_bertscore_f1",
        "avg_clipscore",
        "avg_latency_ms",
        "ollama_usage_rate",
        "fallback_rate",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for s in summaries:
            writer.writerow(
                {
                    "embedding_model": s["embedding_model"],
                    "vector_subdir": s["vector_subdir"],
                    "generation_model": s["generation_model"],
                    "experiment_name": s["experiment"]["name"],
                    "pipeline": s["experiment"]["pipeline"],
                    "mode": s["experiment"]["mode"],
                    "top_k": s["experiment"]["top_k"],
                    "text_weight": s["experiment"]["text_weight"],
                    "image_weight": s["experiment"]["image_weight"],
                    "n_cases": s["n_cases"],
                    "avg_recall_at_k": s.get("avg_recall_at_k"),
                    "avg_precision_at_k": s.get("avg_precision_at_k"),
                    "avg_f1_retrieval": s.get("avg_f1_retrieval"),
                    "avg_bleu": s.get("avg_bleu"),
                    "avg_rouge_l": s.get("avg_rouge_l"),
                    "avg_bertscore_f1": s.get("avg_bertscore_f1"),
                    "avg_clipscore": s.get("avg_clipscore"),
                    "avg_latency_ms": s.get("avg_latency_ms"),
                    "ollama_usage_rate": s.get("ollama_usage_rate"),
                    "fallback_rate": s.get("fallback_rate"),
                }
            )


def _write_question_report(path: Path, question_answers: Dict[str, Any]) -> None:
    lines = [
        "# Full Evaluation Question Report",
        "",
        "This report is generated automatically from run_full_evaluation_suite.py outputs.",
        "",
        "## Answers",
        "",
    ]
    for k, payload in question_answers.items():
        lines.append(f"### {k}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(payload, indent=2))
        lines.append("```")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _summary_filename(embedding_model: str, generation_model: str, experiment_name: str) -> str:
    return (
        f"summary_embed_{_slug(embedding_model)}"
        f"__gen_{_slug(generation_model)}"
        f"__exp_{_slug(experiment_name)}.json"
    )


def _evaluate_embedding_generation_pair(
    root: Path,
    benchmark: List[Dict[str, Any]],
    embedding_model: str,
    vector_subdir: str,
    generation_model: str,
    experiments: List[Experiment],
    temperature: float,
    memory_turn_window: int,
    metric_kwargs: Dict[str, Any],
    bootstrap_samples: int,
    ci_level: float,
    bootstrap_seed: int,
) -> List[Dict[str, Any]]:
    store = MultimodalChromaStore(root=root, model_name=embedding_model, vector_subdir=vector_subdir)
    store.load_index()

    metrics = MetricEngines(**metric_kwargs)

    pair_summaries: List[Dict[str, Any]] = []
    for exp in experiments:
        records: List[Dict[str, Any]] = []
        memory_state: Dict[str, List[str]] = defaultdict(list)
        memory_chat_state: Dict[str, List[Dict[str, str]]] = defaultdict(list)

        agent = PlantMultimodalAgent(
            store=store,
            generation=GenerationConfig(
                use_llm=True,
                model_name=generation_model,
                temperature=temperature,
            ),
            enable_query_rewrite=exp.enable_query_rewrite,
            enable_query_summarize=exp.enable_query_summarize,
            max_memory_turns=max(1, int(memory_turn_window)),
        )

        bert_candidates: List[str] = []
        bert_references: List[str] = []
        bert_record_indexes: List[int] = []

        for case in benchmark:
            raw_query = str(case.get("query", "")).strip()
            if not raw_query:
                continue

            query_image = _resolve_image(root, case.get("query_image"))
            expected_labels = list(case.get("expected_labels", []))
            reference_answer = str(case.get("reference_answer", "")).strip()
            family = str(case.get("family", "unknown"))
            thread_id = case.get("thread_id")
            turn_id = case.get("turn_id")

            query_for_run = raw_query
            if exp.pipeline == "rag_memory":
                query_for_run = _build_memory_query(
                    raw_query,
                    None if thread_id is None else str(thread_id),
                    memory_state,
                    memory_turn_window,
                )
            elif exp.pipeline == "rag_toggle" and exp.enable_memory:
                query_for_run = _build_memory_query(
                    raw_query,
                    None if thread_id is None else str(thread_id),
                    memory_state,
                    memory_turn_window,
                )

            base_cfg = RetrievalConfig(
                mode=exp.mode,
                top_k=exp.top_k,
                text_weight=exp.text_weight,
                image_weight=exp.image_weight,
                text_threshold=0.05,
                image_threshold=0.10,
            )

            t0 = time.perf_counter()

            retrieved_items: List[Dict[str, Any]] = []
            answer = ""
            generation_mode_used = "none"
            generation_model_used = "none"
            generation_note = ""
            route_used = "unknown"
            route_reason = ""
            rewritten_query = query_for_run
            summarized_query = query_for_run

            if exp.pipeline == "generation_only_no_index":
                out = _invoke_generation_only(
                    query=query_for_run,
                    query_image=query_image,
                    generation_model=generation_model,
                    temperature=temperature,
                )
                answer = out.get("answer", "")
                generation_mode_used = out.get("generation_mode_used", "failed")
                generation_model_used = out.get("generation_model_used", generation_model)
                generation_note = out.get("generation_note", "")

            elif exp.pipeline == "retrieval_only":
                retrieved_items = store.search(query_for_run, query_image, base_cfg)
                answer = ""
                generation_mode_used = "none"
                generation_model_used = "none"
                generation_note = "retrieval-only pipeline"

            elif exp.pipeline == "rag_fixed":
                out = agent.invoke(query_for_run, query_image, base_cfg)
                retrieved_items = out.get("retrieved_items", [])
                answer = out.get("answer", "")
                generation_mode_used = out.get("generation_mode_used", "unknown")
                generation_model_used = out.get("generation_model_used", "unknown")
                generation_note = out.get("generation_note", "")
                route_used = "retrieval" if out.get("use_retrieval", True) else "memory"
                route_reason = out.get("retrieval_decision_reason", "")
                rewritten_query = out.get("rewritten_query", query_for_run)
                summarized_query = out.get("summarized_query", query_for_run)

            elif exp.pipeline == "rag_routed":
                routed_cfg = _route_config(base_cfg, case)
                out = agent.invoke(query_for_run, query_image, routed_cfg)
                retrieved_items = out.get("retrieved_items", [])
                answer = out.get("answer", "")
                generation_mode_used = out.get("generation_mode_used", "unknown")
                generation_model_used = out.get("generation_model_used", "unknown")
                generation_note = out.get("generation_note", "")
                route_used = "retrieval" if out.get("use_retrieval", True) else "memory"
                route_reason = out.get("retrieval_decision_reason", "")
                rewritten_query = out.get("rewritten_query", query_for_run)
                summarized_query = out.get("summarized_query", query_for_run)

            elif exp.pipeline == "rag_memory":
                routed_cfg = _route_config(base_cfg, case)
                out = agent.invoke(query_for_run, query_image, routed_cfg)
                retrieved_items = out.get("retrieved_items", [])
                answer = out.get("answer", "")
                generation_mode_used = out.get("generation_mode_used", "unknown")
                generation_model_used = out.get("generation_model_used", "unknown")
                generation_note = out.get("generation_note", "")
                route_used = "retrieval" if out.get("use_retrieval", True) else "memory"
                route_reason = out.get("retrieval_decision_reason", "")
                rewritten_query = out.get("rewritten_query", query_for_run)
                summarized_query = out.get("summarized_query", query_for_run)

                if thread_id is not None:
                    tid = str(thread_id)
                    memory_state[tid].append(f"User: {raw_query}")
                    memory_state[tid].append(f"Assistant: {answer[:300]}")

            elif exp.pipeline == "rag_toggle":
                if not exp.enable_retrieval:
                    out = _invoke_generation_only(
                        query=query_for_run,
                        query_image=query_image,
                        generation_model=generation_model,
                        temperature=temperature,
                    )
                    answer = out.get("answer", "")
                    generation_mode_used = out.get("generation_mode_used", "failed")
                    generation_model_used = out.get("generation_model_used", generation_model)
                    generation_note = out.get("generation_note", "")
                    route_used = "memory"
                    route_reason = "Retrieval disabled by experiment toggle."
                    rewritten_query = query_for_run
                    summarized_query = query_for_run
                else:
                    chat_history_for_turn: List[Dict[str, str]] = []
                    if exp.enable_memory and thread_id is not None:
                        chat_history_for_turn = list(memory_chat_state.get(str(thread_id), []))

                    run_cfg = _route_config(base_cfg, case) if exp.use_routing else base_cfg
                    out = agent.invoke(
                        query=query_for_run,
                        query_image_path=query_image,
                        retrieval_config=run_cfg,
                        chat_history=chat_history_for_turn,
                    )
                    retrieved_items = out.get("retrieved_items", [])
                    answer = out.get("answer", "")
                    generation_mode_used = out.get("generation_mode_used", "unknown")
                    generation_model_used = out.get("generation_model_used", "unknown")
                    generation_note = out.get("generation_note", "")
                    route_used = "retrieval" if out.get("use_retrieval", True) else "memory"
                    route_reason = out.get("retrieval_decision_reason", "")
                    rewritten_query = out.get("rewritten_query", query_for_run)
                    summarized_query = out.get("summarized_query", query_for_run)

                if exp.enable_memory and thread_id is not None:
                    tid = str(thread_id)
                    memory_state[tid].append(f"User: {raw_query}")
                    memory_state[tid].append(f"Assistant: {answer[:300]}")
                    memory_chat_state[tid].append({"role": "user", "content": raw_query})
                    memory_chat_state[tid].append({"role": "assistant", "content": answer[:600]})

            else:
                raise ValueError(f"Unknown pipeline: {exp.pipeline}")

            latency_ms = (time.perf_counter() - t0) * 1000.0

            retrieved_labels = [x.get("label", "") for x in retrieved_items]
            precision = _precision_at_k(retrieved_labels, expected_labels)
            recall = _recall_at_k(retrieved_labels, expected_labels)
            f1 = _f1_score(precision, recall)

            bleu = None
            rouge1 = None
            rouge_l = None
            clipscore = None

            if answer and reference_answer:
                bleu = metrics.bleu(answer, reference_answer)
                rouge1, rouge_l = metrics.rouge(answer, reference_answer)

                img_for_clip = _clipscore_image_for_record(root, query_image, retrieved_items)
                clipscore = metrics.clipscore(answer, img_for_clip)

                bert_record_indexes.append(len(records))
                bert_candidates.append(answer)
                bert_references.append(reference_answer)

            record = {
                "id": case.get("id", ""),
                "family": family,
                "thread_id": thread_id,
                "turn_id": turn_id,
                "query": raw_query,
                "query_used": query_for_run,
                "rewritten_query": rewritten_query,
                "summarized_query": summarized_query,
                "route_used": route_used,
                "route_reason": route_reason,
                "query_image": query_image,
                "expected_labels": expected_labels,
                "retrieved_labels": retrieved_labels,
                "top_label": retrieved_labels[0] if retrieved_labels else "",
                "answer": answer,
                "reference_answer": reference_answer,
                "generation_mode_used": generation_mode_used,
                "generation_model_used": generation_model_used,
                "generation_note": generation_note,
                "precision_at_k": precision,
                "recall_at_k": recall,
                "f1_retrieval": f1,
                "bleu": bleu,
                "rouge1": rouge1,
                "rouge_l": rouge_l,
                "bertscore_f1": None,
                "clipscore": clipscore,
                "latency_ms": latency_ms,
            }
            records.append(record)

        if bert_record_indexes:
            bert_scores = metrics.bertscore_batch(bert_candidates, bert_references)
            for idx, score in zip(bert_record_indexes, bert_scores):
                records[idx]["bertscore_f1"] = score

        summary = _summarize_records(
            records=records,
            embedding_model=embedding_model,
            vector_subdir=vector_subdir,
            generation_model=generation_model,
            experiment=exp,
            bootstrap_samples=bootstrap_samples,
            ci_level=ci_level,
            bootstrap_seed=bootstrap_seed,
        )

        print(
            f"[DONE] {exp.name} | embed={embedding_model} | gen={generation_model} "
            f"| recall={summary.get('avg_recall_at_k')} "
            f"| bleu={summary.get('avg_bleu')} "
            f"| rougeL={summary.get('avg_rouge_l')} "
            f"| bert={summary.get('avg_bertscore_f1')} "
            f"| clip={summary.get('avg_clipscore')}"
        )
        pair_summaries.append(summary)

    return pair_summaries


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()

    benchmark_path = Path(args.benchmark)
    if not benchmark_path.is_absolute():
        benchmark_path = root / benchmark_path
    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmark: List[Dict[str, Any]] = json.load(f)

    if args.max_cases > 0:
        benchmark = benchmark[: args.max_cases]

    embedding_models = _parse_csv(args.embedding_models)
    vector_subdirs = _parse_csv(args.vector_subdirs)
    generation_models = _parse_csv(args.generation_models)
    top_ks = _parse_int_csv(args.top_k_values)
    weight_pairs = _parse_weight_pairs(args.weight_pairs)

    if len(embedding_models) != len(vector_subdirs):
        raise ValueError("embedding-models and vector-subdirs must have the same number of entries")

    save_dir = Path(args.save_dir)
    if not save_dir.is_absolute():
        save_dir = root / save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.parallel_jobs < 1:
        raise ValueError("--parallel-jobs must be >= 1")
    if args.bootstrap_samples < 1:
        raise ValueError("--bootstrap-samples must be >= 1")
    if not (0.0 < args.ci_level < 1.0):
        raise ValueError("--ci-level must be in (0, 1)")

    metric_kwargs = {
        "clipscore_model": args.clipscore_model,
        "disable_bleu": args.disable_bleu,
        "disable_rouge": args.disable_rouge,
        "disable_bertscore": args.disable_bertscore,
        "disable_clipscore": args.disable_clipscore,
        "bertscore_model": args.bertscore_model,
    }

    metrics_probe = MetricEngines(**metric_kwargs)
    support_path = save_dir / "metric_support.json"
    _save_summary(support_path, metrics_probe.support)

    experiments = _build_experiments(
        top_ks,
        weight_pairs,
        include_toggle_ablations=not args.disable_toggle_ablations,
    )
    all_summaries: List[Dict[str, Any]] = []

    pair_jobs: List[Tuple[str, str, str]] = []
    for embedding_model, vector_subdir in zip(embedding_models, vector_subdirs):
        for generation_model in generation_models:
            pair_jobs.append((embedding_model, vector_subdir, generation_model))

    worker_count = min(args.parallel_jobs, max(1, len(pair_jobs)))
    print(f"Running {len(pair_jobs)} embedding-generation jobs with parallel-jobs={worker_count}")

    if worker_count > 1 and len(pair_jobs) > 1:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_to_job = {}
            for embedding_model, vector_subdir, generation_model in pair_jobs:
                future = executor.submit(
                    _evaluate_embedding_generation_pair,
                    root,
                    benchmark,
                    embedding_model,
                    vector_subdir,
                    generation_model,
                    experiments,
                    args.temperature,
                    args.memory_turn_window,
                    metric_kwargs,
                    args.bootstrap_samples,
                    args.ci_level,
                    args.bootstrap_seed,
                )
                future_to_job[future] = (embedding_model, vector_subdir, generation_model)

            for future in as_completed(future_to_job):
                embedding_model, vector_subdir, generation_model = future_to_job[future]
                try:
                    pair_summaries = future.result()
                except Exception as exc:
                    raise RuntimeError(
                        f"Parallel job failed for embed={embedding_model}, vector_subdir={vector_subdir}, gen={generation_model}"
                    ) from exc

                for summary in pair_summaries:
                    out_name = _summary_filename(
                        embedding_model=summary["embedding_model"],
                        generation_model=summary["generation_model"],
                        experiment_name=summary["experiment"]["name"],
                    )
                    _save_summary(save_dir / out_name, summary)
                all_summaries.extend(pair_summaries)
                print(
                    f"[PAIR DONE] embed={embedding_model} | gen={generation_model} "
                    f"| experiments={len(pair_summaries)}"
                )
    else:
        for embedding_model, vector_subdir, generation_model in pair_jobs:
            pair_summaries = _evaluate_embedding_generation_pair(
                root=root,
                benchmark=benchmark,
                embedding_model=embedding_model,
                vector_subdir=vector_subdir,
                generation_model=generation_model,
                experiments=experiments,
                temperature=args.temperature,
                memory_turn_window=args.memory_turn_window,
                metric_kwargs=metric_kwargs,
                bootstrap_samples=args.bootstrap_samples,
                ci_level=args.ci_level,
                bootstrap_seed=args.bootstrap_seed,
            )
            for summary in pair_summaries:
                out_name = _summary_filename(
                    embedding_model=summary["embedding_model"],
                    generation_model=summary["generation_model"],
                    experiment_name=summary["experiment"]["name"],
                )
                _save_summary(save_dir / out_name, summary)
            all_summaries.extend(pair_summaries)

    all_summaries.sort(
        key=lambda s: (
            s.get("embedding_model", ""),
            s.get("vector_subdir", ""),
            s.get("generation_model", ""),
            s.get("experiment", {}).get("name", ""),
        )
    )

    matrix = {
        "benchmark": str(benchmark_path),
        "n_runs": len(all_summaries),
        "ci_config": {
            "level": args.ci_level,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
        },
        "parallel_jobs_used": worker_count,
        "runs": [
            {
                "embedding_model": s["embedding_model"],
                "vector_subdir": s["vector_subdir"],
                "generation_model": s["generation_model"],
                "experiment": s["experiment"],
                "avg_recall_at_k": s.get("avg_recall_at_k"),
                "avg_bleu": s.get("avg_bleu"),
                "avg_rouge_l": s.get("avg_rouge_l"),
                "avg_bertscore_f1": s.get("avg_bertscore_f1"),
                "avg_clipscore": s.get("avg_clipscore"),
                "ollama_usage_rate": s.get("ollama_usage_rate"),
                "fallback_rate": s.get("fallback_rate"),
                "avg_latency_ms": s.get("avg_latency_ms"),
            }
            for s in all_summaries
        ],
    }

    question_answers = _build_question_answers(
        all_summaries,
        bootstrap_samples=args.bootstrap_samples,
        ci_level=args.ci_level,
        bootstrap_seed=args.bootstrap_seed,
    )

    _save_summary(save_dir / "summary_all_runs.json", matrix)
    _save_summary(save_dir / "question_answers.json", question_answers)
    _write_question_report(save_dir / "question_answers.md", question_answers)
    _write_csv(save_dir / "leaderboard.csv", all_summaries)

    print(f"Saved run matrix to: {save_dir / 'summary_all_runs.json'}")
    print(f"Saved leaderboard CSV to: {save_dir / 'leaderboard.csv'}")
    print(f"Saved question report to: {save_dir / 'question_answers.md'}")


if __name__ == "__main__":
    main()
