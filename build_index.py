from __future__ import annotations

import argparse
from pathlib import Path
import csv
import json
import shutil
import os
from typing import Optional, Tuple

try:
    from .vector_store import DEFAULT_MODEL, MultimodalChromaStore, SUPPORTED_MODELS
except ImportError:
    from vector_store import DEFAULT_MODEL, MultimodalChromaStore, SUPPORTED_MODELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Chroma index for Pokemon RAG")
    parser.add_argument("--root", default=".", help="Project root containing this folder")
    parser.add_argument(
        "--dataset-root",
        default="4POISONEDIMAGE",
        help="Path to image dataset root (default: 4POISONEDIMAGE)",
    )
    parser.add_argument(
        "--text-csv",
        default="4POISONEDCONFLICT/pokemon.csv",
        help="Optional CSV file describing images and labels (relative to root).",
    )
    parser.add_argument(
        "--cure-json",
        default="samples/cure.json",
        help="Path to cure knowledge JSON to enrich text encoder/database",
    )
    parser.add_argument(
        "--max-images-per-class",
        type=int,
        default=10,
        help="Maximum number of images to index per class folder for quick testing",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL,
        help=(
            "Embedding model to use. Examples: openai/clip-vit-base-patch16, "
            "openai/clip-vit-base-patch32"
        ),
    )
    parser.add_argument(
        "--vector-subdir",
        default="vector_dbPOISONEDIMAGE/chroma",
        help="Relative output directory for the index, e.g. vector_db/chroma_clip16",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing vector DB")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()

    if args.model_name.strip().lower() not in SUPPORTED_MODELS:
        raise ValueError(
            "Unknown or unsupported model: "
            f"{args.model_name}. Supported models: {', '.join(sorted(SUPPORTED_MODELS))}"
        )

    store = MultimodalChromaStore(
        root=root,
        model_name=args.model_name,
        vector_subdir=args.vector_subdir,
    )

    # If a CSV is provided, prepare a temporary per-label folder structure
    dataset_path, cure_json = prepare_dataset_for_index(
        root=root,
        dataset_root=args.dataset_root,
        text_csv=args.text_csv,
    )

    cure_json_path = cure_json
    if cure_json_path is None:
        default_cure_json = Path(args.cure_json)
        if default_cure_json.is_absolute():
            candidate = default_cure_json
        else:
            candidate = (root / default_cure_json).resolve()
        if candidate.exists():
            cure_json_path = str(candidate)

    store.build_index(
        dataset_root=dataset_path,
        overwrite=args.overwrite,
        cure_json_path=cure_json_path,
        max_images_per_class=args.max_images_per_class,
    )

    print("Indexing complete.")
    print(f"Vector DB: {store.vector_dir}")
    print(f"Model used: {store.model_name}")
    print(f"Records indexed: {len(store.records)}")


def _detect_column(header: list[str], choices: list[str]) -> Optional[str]:
    low = [h.strip().lower() for h in header]
    for c in choices:
        if c in low:
            return header[low.index(c)]
    return None


def prepare_dataset_for_index(root: Path, dataset_root: str | Path, text_csv: Optional[str | Path]) -> Tuple[Path, Optional[Path]]:
    """
    Prepare a per-label directory layout for the index. If `text_csv` is provided and exists,
    this will create a temporary folder under `root/.tmp_index_dataset` where images are
    organized into label subfolders according to the CSV. The function also writes a
    small cure JSON mapping file (label -> description) and returns its path.

    Returns: (dataset_root_path, cure_json_path_or_None)
    """
    dataset_root = Path(dataset_root)
    # Determine where images live. Common layouts: <root>/<dataset_root>/images or <root>/<dataset_root>
    candidate_img_dir = (root / dataset_root / "images").resolve()
    if candidate_img_dir.exists():
        images_base = candidate_img_dir
    else:
        images_base = (root / dataset_root).resolve()

    if not text_csv:
        return (root / dataset_root, None)

    csv_path = Path(text_csv)
    if not csv_path.is_absolute():
        csv_path = (root / csv_path).resolve()

    if not csv_path.exists():
        # No CSV; return original dataset path and no cure json
        return (root / dataset_root, None)

    tmp_root = (root / ".tmp_index_dataset").resolve()
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)

    cure_map: dict[str, str] = {}
    # Attempt to find likely column names
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or [])
        img_col = _detect_column(header, ["image", "filename", "file", "file_name", "image_name"]) or (header[0] if header else "image")
        label_col = _detect_column(header, ["name", "label", "class", "type"]) or (header[1] if len(header) > 1 else "label")
        text_col = _detect_column(header, ["description", "text", "evolution"]) if header else None

        for row in reader:
            img_name = str(row.get(img_col, "")).strip()
            label = str(row.get(label_col, "")).strip() or "unknown"
            desc = str(row.get(text_col, "")).strip() if text_col else ""

            if not img_name:
                continue

            src = (images_base / img_name).resolve()
            if not src.exists():
                # try without folders
                alt = (images_base / Path(img_name).name).resolve()
                if alt.exists():
                    src = alt
                else:
                    base_name = Path(img_name).name
                    if not Path(base_name).suffix:
                        for ext in [".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"]:
                            alt_ext = (images_base / (base_name + ext)).resolve()
                            if alt_ext.exists():
                                src = alt_ext
                                break
                    if not src.exists():
                        continue

            dst_dir = tmp_root / label
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / src.name
            try:
                # create a symlink when possible to save space
                if not dst.exists():
                    dst.symlink_to(str(src))
            except Exception:
                # fallback to copy
                if not dst.exists():
                    shutil.copy2(src, dst)

            if desc:
                # keep first non-empty description per label
                if label not in cure_map or not cure_map[label]:
                    cure_map[label] = desc

    cure_json_path: Optional[Path] = None
    if cure_map:
        cure_json_path = tmp_root / "cure_map.json"
        with open(cure_json_path, "w", encoding="utf-8") as f:
            json.dump(cure_map, f, indent=2)

    return (tmp_root, cure_json_path)


if __name__ == "__main__":
    main()
