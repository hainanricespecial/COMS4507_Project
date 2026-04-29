# run_all_ablation.py
"""
Script to run all ablation studies in sequence and evaluate them.
- Iterates over retrieval modes and key parameter settings
- Runs eval_ablation.py for each config
- Saves results to ablation_results/
"""
import os
import subprocess
import json
from pathlib import Path

MODES = ["text_only", "image_only", "hybrid", "auto"]
TOP_K = [5]
TEXT_WEIGHTS = [0.5, 0.7, 0.3]  # Only for hybrid/auto
IMAGE_WEIGHTS = [0.5, 0.3, 0.7]  # Only for hybrid/auto
BENCHMARK = "benchmark_queries.json"
RESULTS_DIR = "ablation_results"

os.makedirs(RESULTS_DIR, exist_ok=True)

root = Path(__file__).parent.resolve()

for mode in MODES:
    for top_k in TOP_K:
        if mode in ("hybrid", "auto"):
            for tw, iw in zip(TEXT_WEIGHTS, IMAGE_WEIGHTS):
                out_json = f"{RESULTS_DIR}/result_{mode}_tk{top_k}_tw{tw}_iw{iw}.json"
                cmd = [
                    "python", "eval_ablation.py",
                    "--root", str(root),
                    "--benchmark", BENCHMARK,
                    "--mode", mode,
                    "--top-k", str(top_k),
                    "--text-weight", str(tw),
                    "--image-weight", str(iw),
                    "--save-json", out_json
                ]
                print(f"Running: {' '.join(cmd)})")
                subprocess.run(cmd, check=True)
        else:
            out_json = f"{RESULTS_DIR}/result_{mode}_tk{top_k}.json"
            cmd = [
                "python", "eval_ablation.py",
                "--root", str(root),
                "--benchmark", BENCHMARK,
                "--mode", mode,
                "--top-k", str(top_k),
                "--save-json", out_json
            ]
            print(f"Running: {' '.join(cmd)})")
            subprocess.run(cmd, check=True)

print("All ablation runs complete. Results saved in ablation_results/")
