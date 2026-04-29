from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean
from typing import Any
import math
try:
    from .vector_store import MultimodalChromaStore, RetrievalConfig
except ImportError:
    from vector_store import MultimodalChromaStore, RetrievalConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retrieval ablation on benchmark queries")
    parser.add_argument("--root", default=".")
    parser.add_argument("--benchmark", default="benchmark_queries.json")
    parser.add_argument("--mode", default="hybrid", choices=["text_only", "image_only", "hybrid", "auto"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--text-weight", type=float, default=0.5)
    parser.add_argument("--image-weight", type=float, default=0.5)
    parser.add_argument("--text-threshold", type=float, default=0.05)
    parser.add_argument("--image-threshold", type=float, default=0.10)
    parser.add_argument("--save-json", default="")
    return parser.parse_args()



# --- Metrics ---
def recall_at_k(results: list[dict[str, Any]], expected_labels: list[str]) -> float:
    if not expected_labels:
        return 0.0
    got = {x["label"].lower() for x in results}
    exp = {x.lower() for x in expected_labels}
    return 1.0 if got.intersection(exp) else 0.0

def precision_at_k(results: list[dict[str, Any]], expected_labels: list[str]) -> float:
    if not results:
        return 0.0
    got = [x["label"].lower() for x in results]
    exp = set(x.lower() for x in expected_labels)
    hits = sum([1 for label in got if label in exp])
    return hits / len(got)

def f1_at_k(results: list[dict[str, Any]], expected_labels: list[str]) -> float:
    p = precision_at_k(results, expected_labels)
    r = recall_at_k(results, expected_labels)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)

def ndcg_at_k(results: list[dict[str, Any]], expected_labels: list[str]) -> float:
    exp = set(x.lower() for x in expected_labels)
    dcg = 0.0
    for i, x in enumerate(results):
        if x["label"].lower() in exp:
            dcg += 1.0 / (math.log2(i + 2))
    # Ideal DCG
    ideal_hits = min(len(exp), len(results))
    idcg = sum(1.0 / (math.log2(i + 2)) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0

def mrr_at_k(results: list[dict[str, Any]], expected_labels: list[str]) -> float:
    exp = set(x.lower() for x in expected_labels)
    for i, x in enumerate(results):
        if x["label"].lower() in exp:
            return 1.0 / (i + 1)
    return 0.0

def map_at_k(results: list[dict[str, Any]], expected_labels: list[str]) -> float:
    exp = set(x.lower() for x in expected_labels)
    hits = 0
    sum_precisions = 0.0
    for i, x in enumerate(results):
        if x["label"].lower() in exp:
            hits += 1
            sum_precisions += hits / (i + 1)
    return sum_precisions / len(exp) if exp else 0.0



def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()

    benchmark_path = Path(args.benchmark)
    if not benchmark_path.is_absolute():
        benchmark_path = root / benchmark_path

    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    cfg = RetrievalConfig(
        mode=args.mode,
        top_k=args.top_k,
        text_weight=args.text_weight,
        image_weight=args.image_weight,
        text_threshold=args.text_threshold,
        image_threshold=args.image_threshold,
    )

    store = MultimodalChromaStore(root=root)
    store.load_index()


    import math
    records = []
    recalls = []
    precisions = []
    f1s = []
    ndcgs = []
    mrrs = []
    maps = []
    latencies_ms = []

    for case in benchmark:
        query = case["query"]
        expected = case.get("expected_labels", [])
        query_image = case.get("query_image")
        if query_image:
            p = Path(query_image)
            query_image = str(p if p.is_absolute() else (root / p))

        t0 = time.perf_counter()
        out = store.search(query, query_image, cfg)
        latency_ms = (time.perf_counter() - t0) * 1000


        rec = recall_at_k(out, expected)
        prec = precision_at_k(out, expected)
        f1 = f1_at_k(out, expected)
        ndcg = ndcg_at_k(out, expected)
        mrr = mrr_at_k(out, expected)
        mapk = map_at_k(out, expected)
        recalls.append(rec)
        precisions.append(prec)
        f1s.append(f1)
        ndcgs.append(ndcg)
        mrrs.append(mrr)
        maps.append(mapk)
        latencies_ms.append(latency_ms)

        records.append(
            {
                "query": query,
                "expected_labels": expected,
                "retrieved_labels": [x["label"] for x in out],
                "recall_at_k": rec,
                "precision_at_k": prec,
                "f1_at_k": f1,
                "ndcg_at_k": ndcg,
                "mrr_at_k": mrr,
                "map_at_k": mapk,
                "latency_ms": latency_ms,
            }
        )


    summary = {
        "mode": cfg.mode,
        "top_k": cfg.top_k,
        "text_weight": cfg.text_weight,
        "image_weight": cfg.image_weight,
        "text_threshold": cfg.text_threshold,
        "image_threshold": cfg.image_threshold,
        "n_queries": len(records),
        "avg_recall_at_k": mean(recalls) if recalls else 0.0,
        "avg_precision_at_k": mean(precisions) if precisions else 0.0,
        "avg_f1_at_k": mean(f1s) if f1s else 0.0,
        "avg_ndcg_at_k": mean(ndcgs) if ndcgs else 0.0,
        "avg_mrr_at_k": mean(mrrs) if mrrs else 0.0,
        "avg_map_at_k": mean(maps) if maps else 0.0,
        "avg_latency_ms": mean(latencies_ms) if latencies_ms else 0.0,
        "records": records,
    }

    print(json.dumps(summary, indent=2))

    if args.save_json:
        save_path = Path(args.save_json)
        if not save_path.is_absolute():
            save_path = root / save_path
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
