from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from statistics import mean
from typing import Any

try:
    from .local_agent import GenerationConfig, PokemonMultimodalAgent
    from .vector_store import MultimodalChromaStore, RetrievalConfig
except ImportError:
    from local_agent import GenerationConfig, PokemonMultimodalAgent
    from vector_store import MultimodalChromaStore, RetrievalConfig


def _parse_csv(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _norm_text(raw: str) -> str:
    s = raw.lower().replace("_", " ").replace("-", " ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _safe_name(raw: str) -> str:
    s = raw.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def _recall_at_k(results: list[dict[str, Any]], expected_labels: list[str]) -> float:
    if not expected_labels:
        return 0.0
    got = {x["label"].lower() for x in results}
    exp = {x.lower() for x in expected_labels}
    return 1.0 if got.intersection(exp) else 0.0


def _answer_label_hit(answer: str, expected_labels: list[str]) -> float:
    if not expected_labels:
        return 0.0
    answer_norm = _norm_text(answer)
    for label in expected_labels:
        label_norm = _norm_text(label)
        if label_norm and label_norm in answer_norm:
            return 1.0
    return 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run 4-combo ablation over embedding models x generation models "
            "using benchmark queries."
        )
    )
    parser.add_argument("--root", default=".")
    parser.add_argument("--benchmark", default="benchmark_queries.json")
    parser.add_argument("--mode", default="hybrid", choices=["text_only", "image_only", "hybrid", "auto"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--text-weight", type=float, default=0.5)
    parser.add_argument("--image-weight", type=float, default=0.5)
    parser.add_argument("--text-threshold", type=float, default=0.05)
    parser.add_argument("--image-threshold", type=float, default=0.10)
    parser.add_argument(
        "--embedding-models",
        default="openai/clip-vit-base-patch16,openai/clip-vit-base-patch32",
        help="Comma-separated embedding model list.",
    )
    parser.add_argument(
        "--vector-subdirs",
        default="vector_db/chroma_clip16,vector_db/chroma_clip32",
        help="Comma-separated vector subdirs aligned with embedding-models.",
    )
    parser.add_argument(
        "--generation-models",
        default="qwen3.5:0.8b,qwen3.5:2b",
        help="Comma-separated Ollama generation model list.",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--save-dir", default="ablation_results/generation_model_ablation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()

    benchmark_path = Path(args.benchmark)
    if not benchmark_path.is_absolute():
        benchmark_path = root / benchmark_path

    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    embedding_models = _parse_csv(args.embedding_models)
    vector_subdirs = _parse_csv(args.vector_subdirs)
    generation_models = _parse_csv(args.generation_models)

    if len(embedding_models) != len(vector_subdirs):
        raise ValueError(
            "embedding-models and vector-subdirs must have equal length. "
            f"Got {len(embedding_models)} and {len(vector_subdirs)}"
        )

    if not embedding_models:
        raise ValueError("No embedding models provided.")
    if not generation_models:
        raise ValueError("No generation models provided.")

    cfg = RetrievalConfig(
        mode=args.mode,
        top_k=args.top_k,
        text_weight=args.text_weight,
        image_weight=args.image_weight,
        text_threshold=args.text_threshold,
        image_threshold=args.image_threshold,
    )

    save_dir = Path(args.save_dir)
    if not save_dir.is_absolute():
        save_dir = root / save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    matrix: list[dict[str, Any]] = []

    for embedding_model, vector_subdir in zip(embedding_models, vector_subdirs):
        store = MultimodalChromaStore(
            root=root,
            model_name=embedding_model,
            vector_subdir=vector_subdir,
        )
        store.load_index()

        for generation_model in generation_models:
            agent = PokemonMultimodalAgent(
                store=store,
                generation=GenerationConfig(
                    use_llm=True,
                    model_name=generation_model,
                    temperature=args.temperature,
                ),
            )

            records = []
            recalls = []
            answer_hits = []
            latencies_ms = []
            ollama_used_flags = []
            fallback_flags = []

            for case in benchmark:
                query = case["query"]
                expected = case.get("expected_labels", [])
                query_image = case.get("query_image")
                if query_image:
                    p = Path(query_image)
                    query_image = str(p if p.is_absolute() else (root / p))

                t0 = time.perf_counter()
                out = agent.invoke(query, query_image, cfg)
                latency_ms = (time.perf_counter() - t0) * 1000.0

                retrieved = out.get("retrieved_items", [])
                answer = out.get("answer", "")
                generation_mode_used = out.get("generation_mode_used", "unknown")
                generation_model_used = out.get("generation_model_used", "unknown")
                generation_note = out.get("generation_note", "")

                rec = _recall_at_k(retrieved, expected)
                ans_hit = _answer_label_hit(answer, expected)
                used_ollama = 1.0 if generation_mode_used == "ollama" else 0.0
                fallback = 1.0 if generation_mode_used != "ollama" else 0.0

                recalls.append(rec)
                answer_hits.append(ans_hit)
                latencies_ms.append(latency_ms)
                ollama_used_flags.append(used_ollama)
                fallback_flags.append(fallback)

                records.append(
                    {
                        "family": case.get("family", ""),
                        "query": query,
                        "expected_labels": expected,
                        "retrieved_labels": [x["label"] for x in retrieved],
                        "top_label": retrieved[0]["label"] if retrieved else "",
                        "recall_at_k": rec,
                        "answer_label_hit": ans_hit,
                        "generation_mode_used": generation_mode_used,
                        "generation_model_used": generation_model_used,
                        "generation_note": generation_note,
                        "latency_ms": latency_ms,
                    }
                )

            summary = {
                "embedding_model": embedding_model,
                "vector_subdir": vector_subdir,
                "generation_model_requested": generation_model,
                "mode": cfg.mode,
                "top_k": cfg.top_k,
                "text_weight": cfg.text_weight,
                "image_weight": cfg.image_weight,
                "text_threshold": cfg.text_threshold,
                "image_threshold": cfg.image_threshold,
                "n_queries": len(records),
                "avg_recall_at_k": mean(recalls) if recalls else 0.0,
                "avg_answer_label_hit": mean(answer_hits) if answer_hits else 0.0,
                "ollama_usage_rate": mean(ollama_used_flags) if ollama_used_flags else 0.0,
                "fallback_rate": mean(fallback_flags) if fallback_flags else 0.0,
                "avg_latency_ms": mean(latencies_ms) if latencies_ms else 0.0,
                "records": records,
            }

            emb_name = _safe_name(embedding_model)
            gen_name = _safe_name(generation_model)
            combo_path = save_dir / f"result_embed_{emb_name}__gen_{gen_name}.json"
            with open(combo_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

            matrix.append(
                {
                    "embedding_model": embedding_model,
                    "vector_subdir": vector_subdir,
                    "generation_model_requested": generation_model,
                    "result_file": str(combo_path),
                    "avg_recall_at_k": summary["avg_recall_at_k"],
                    "avg_answer_label_hit": summary["avg_answer_label_hit"],
                    "ollama_usage_rate": summary["ollama_usage_rate"],
                    "fallback_rate": summary["fallback_rate"],
                    "avg_latency_ms": summary["avg_latency_ms"],
                }
            )

            print(
                f"[DONE] embed={embedding_model} gen={generation_model} "
                f"recall={summary['avg_recall_at_k']:.3f} "
                f"answer_hit={summary['avg_answer_label_hit']:.3f} "
                f"ollama_usage={summary['ollama_usage_rate']:.3f}"
            )

    matrix_summary = {
        "benchmark": str(benchmark_path),
        "n_combinations": len(matrix),
        "combinations": matrix,
    }
    matrix_path = save_dir / "summary_4combos.json"
    with open(matrix_path, "w", encoding="utf-8") as f:
        json.dump(matrix_summary, f, indent=2)

    print(f"Saved matrix summary: {matrix_path}")


if __name__ == "__main__":
    main()
