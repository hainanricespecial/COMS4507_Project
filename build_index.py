from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .vector_store import DEFAULT_MODEL, MultimodalChromaStore, SUPPORTED_MODELS
except ImportError:
    from vector_store import DEFAULT_MODEL, MultimodalChromaStore, SUPPORTED_MODELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Chroma index for plant multimodal RAG")
    parser.add_argument("--root", default=".", help="Project root containing this folder")
    parser.add_argument(
        "--dataset-root",
        default="../17/Image Data base/Image Data base",
        help="Path to image dataset root (default: ../17/Image Data base/Image Data base)",
    )
    parser.add_argument(
        "--cure-json",
        default="../17/plant diseases cure/cure.json",
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
        default="vector_db/chroma",
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
    store.build_index(
        dataset_root=args.dataset_root,
        overwrite=args.overwrite,
        cure_json_path=args.cure_json,
        max_images_per_class=args.max_images_per_class,
    )

    print("Indexing complete.")
    print(f"Vector DB: {store.vector_dir}")
    print(f"Model used: {store.model_name}")
    print(f"Records indexed: {len(store.records)}")


if __name__ == "__main__":
    main()
