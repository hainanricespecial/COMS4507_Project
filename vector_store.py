from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
import json
import shutil
import re
import sys
import warnings
from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Optional

import chromadb
import numpy as np
import torch
import transformers
from PIL import Image
from dotenv import load_dotenv
from transformers import (
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
    CLIPModel,
    CLIPProcessor,
)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
load_dotenv()


def _repair_invalid_ssl_cert_env() -> None:
    invalid_vars = []
    for name in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        value = os.environ.get(name)
        if value and not Path(value).expanduser().exists():
            invalid_vars.append(name)

    if not invalid_vars:
        return

    try:
        import certifi

        cert_path = certifi.where()
        if not Path(cert_path).exists():
            raise FileNotFoundError(cert_path)
        for name in invalid_vars:
            os.environ[name] = cert_path
        warnings.warn(
            "Replaced invalid certificate environment variable(s) "
            f"{', '.join(invalid_vars)} with certifi bundle: {cert_path}",
            RuntimeWarning,
        )
    except Exception as exc:
        for name in invalid_vars:
            os.environ.pop(name, None)
        warnings.warn(
            "Removed invalid certificate environment variable(s) "
            f"{', '.join(invalid_vars)} because certifi could not be used: {exc}",
            RuntimeWarning,
        )


_repair_invalid_ssl_cert_env()

# becareful with case sensitivity in model names
HF_TOKEN = os.getenv("HF_TOKEN")
DEFAULT_MODEL = os.getenv("HF_MODEL_NAME", "openai/clip-vit-base-patch16").strip()
#DEFAULT_VECTOR_SUBDIR = os.getenv("PMA_VECTOR_SUBDIR", "vector_db/chroma").strip()
DEFAULT_VECTOR_SUBDIR = os.getenv("PMA_VECTOR_SUBDIR", "vector_dbPOISONEDIMAGE/chroma").strip()

SUPPORTED_MODELS = {
    "openai/clip-vit-base-patch32": "clip",
    "openai/clip-vit-base-patch16": "clip",
    "facebook/chameleon-7b": "chameleon",
    "showlab/show-o2-7b": "show-o2",
    "baai/emu3.5": "emu",
    "deepseek-ai/janus-pro-1b": "janus",
    #"lmms-lab/llava-onevision-qwen2-7b-ov": "llava-onevision",
}

CANONICAL_MODEL_IDS = {
    "deepseek-ai/janus-pro-1b": "deepseek-ai/Janus-Pro-1B",
}



@dataclass
class RetrievalConfig:
    mode: str = "hybrid"  # text_only | image_only | hybrid | auto
    top_k: int = 5
    text_weight: float = 0.5
    image_weight: float = 0.5
    text_threshold: float = 0.05
    image_threshold: float = 0.10
    dense_candidate_k: int = 40
    sparse_candidate_k: int = 40
    rrf_k: int = 10
    dense_rrf_weight: float = 1.0
    sparse_rrf_weight: float = 1.0



class MultimodalChromaStore:
    def __init__(
        self,
        root: str | Path,
        model_name: str = DEFAULT_MODEL,
        vector_subdir: str = DEFAULT_VECTOR_SUBDIR,
    ) -> None:
        self.root = Path(root).resolve()
        self.vector_dir = self.root / vector_subdir
        self.metadata_path = self.vector_dir / "metadata.json"
        self.text_collection_name = "pokemon_text"
        self.image_collection_name = "pokemon_image"

        self.requested_model_name = str(model_name).strip()
        self.model_name = self.requested_model_name
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._model_type = None  # 'clip', 'chameleon', 'show-o2', 'emu', etc
        self._last_model_warning: Optional[str] = None

        self.records: list[dict[str, Any]] = []
        self._record_by_id: dict[str, dict[str, Any]] = {}
        self.client: Optional[chromadb.PersistentClient] = None
        self.text_collection: Optional[Any] = None
        self.image_collection: Optional[Any] = None
        self._bm25: Optional[Any] = None
        self._bm25_doc_ids: list[str] = []

    def get_embedding_model_info(self, ensure_loaded: bool = False) -> dict[str, Any]:
        if ensure_loaded and self._model is None:
            try:
                self._lazy_model()
            except Exception as exc:
                return {
                    "requested_model": self.requested_model_name,
                    "active_model": self.model_name,
                    "model_type": self._model_type,
                    "is_loaded": False,
                    "warning": self._last_model_warning,
                    "python_executable": sys.executable,
                    "python_version": sys.version.split()[0],
                    "transformers_version": getattr(transformers, "__version__", "unknown"),
                    "error": str(exc),
                }

        return {
            "requested_model": self.requested_model_name,
            "active_model": self.model_name,
            "model_type": self._model_type,
            "is_loaded": self._model is not None,
            "warning": self._last_model_warning,
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
            "transformers_version": getattr(transformers, "__version__", "unknown"),
            "error": None,
        }

    @staticmethod
    def _parse_major_version(version_text: str) -> Optional[int]:
        m = re.match(r"\s*(\d+)", str(version_text))
        if not m:
            return None
        return int(m.group(1))

    def _janus_runtime_check(self) -> tuple[bool, str]:
        if sys.version_info >= (3, 13):
            return False, f"Python {sys.version.split()[0]} detected"

        tf_version = getattr(transformers, "__version__", "unknown")
        tf_major = self._parse_major_version(tf_version)
        if tf_major is not None and tf_major >= 5:
            return False, f"transformers {tf_version} detected"

        return True, "compatible"

    @staticmethod
    def _safe_token_arg() -> dict[str, str]:
        if HF_TOKEN:
            return {"token": HF_TOKEN}
        return {}

    def _load_janus_runtime(self, model_id: str) -> tuple[Any, Any, Any]:
        """Load Janus via its own runtime package, which registers multi_modality config."""
        try:
            from janus.models import MultiModalityCausalLM, VLChatProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Janus-Pro models need the Janus runtime package. Install with: "
                "pip install git+https://github.com/deepseek-ai/Janus.git"
            ) from exc

        token_kwargs = self._safe_token_arg()
        model = MultiModalityCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            **token_kwargs,
        )
        processor = VLChatProcessor.from_pretrained(
            model_id,
            **token_kwargs,
        )
        tokenizer = getattr(processor, "tokenizer", None)
        return model, processor, tokenizer

    def _load_clip_runtime(self, model_id: str) -> tuple[Any, Any, Any]:
        token_kwargs = self._safe_token_arg()
        model = CLIPModel.from_pretrained(model_id, **token_kwargs)
        model.eval()
        processor = CLIPProcessor.from_pretrained(model_id, **token_kwargs)
        return model, processor, None

    def _lazy_client(self) -> chromadb.PersistentClient:
        if self.client is None:
            self.vector_dir.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(self.vector_dir))
        return self.client

    def _connect_collections(self) -> None:
        client = self._lazy_client()
        self.text_collection = client.get_or_create_collection(
            name=self.text_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.image_collection = client.get_or_create_collection(
            name=self.image_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _lazy_model(self):
        """Load the configured model, processor, and tokenizer once."""
        if self._model is not None and self._processor is not None:
            return self._model, self._processor, self._tokenizer, self._model_type

        self._last_model_warning = None

        key = self.model_name.strip().lower()
        model_type = SUPPORTED_MODELS.get(key)
        if model_type is None:
            raise ValueError(
                f"Unknown or unsupported model: {self.model_name}. Supported models are: {', '.join(sorted(SUPPORTED_MODELS))}"
            )

        resolved_model_id = CANONICAL_MODEL_IDS.get(key, self.model_name.strip())
        self.model_name = resolved_model_id

        token_kwargs = self._safe_token_arg()

        if model_type == "clip":
            self._model, self._processor, self._tokenizer = self._load_clip_runtime(resolved_model_id)
            self._model_type = model_type
            return self._model, self._processor, self._tokenizer, self._model_type

        # The remaining models are multimodal generators; we load them with remote code enabled
        # and pool hidden states into a retrieval vector when possible.
        if model_type == "chameleon":
            self._model = AutoModel.from_pretrained(resolved_model_id, trust_remote_code=True, **token_kwargs)
            self._processor = AutoProcessor.from_pretrained(resolved_model_id, trust_remote_code=True, **token_kwargs)
            self._tokenizer = AutoTokenizer.from_pretrained(resolved_model_id, trust_remote_code=True, **token_kwargs)
        elif model_type == "show-o2":
            self._model = AutoModel.from_pretrained(resolved_model_id, trust_remote_code=True, **token_kwargs)
            self._processor = AutoProcessor.from_pretrained(resolved_model_id, trust_remote_code=True, **token_kwargs)
            self._tokenizer = AutoTokenizer.from_pretrained(resolved_model_id, trust_remote_code=True, **token_kwargs)
        elif model_type == "emu":
            self._model = AutoModel.from_pretrained(resolved_model_id, trust_remote_code=True, **token_kwargs)
            self._processor = AutoProcessor.from_pretrained(resolved_model_id, trust_remote_code=True, **token_kwargs)
            self._tokenizer = AutoTokenizer.from_pretrained(resolved_model_id, trust_remote_code=True, **token_kwargs)
        elif model_type == "janus":
            ok, reason = self._janus_runtime_check()
            if not ok:
                fallback_model = "openai/clip-vit-base-patch16"
                warn_msg = (
                    "Janus-Pro runtime is unavailable/incompatible in this environment "
                    f"({reason}); falling back to {fallback_model}. "
                    "To use Janus-Pro, prefer Python <= 3.12 with a Janus-compatible transformers build."
                )
                warnings.warn(warn_msg)
                self._last_model_warning = warn_msg
                self.model_name = fallback_model
                self._model, self._processor, self._tokenizer = self._load_clip_runtime(fallback_model)
                self._model_type = "clip"
                return self._model, self._processor, self._tokenizer, self._model_type

            try:
                self._model, self._processor, self._tokenizer = self._load_janus_runtime(resolved_model_id)
            except Exception as exc:
                fallback_model = "openai/clip-vit-base-patch16"
                warn_msg = (
                    "Janus-Pro runtime is unavailable/incompatible in this environment; falling back to "
                    f"{fallback_model}. Original error: {exc}. "
                    "To use Janus-Pro, prefer Python <= 3.12 with a Janus-compatible transformers build."
                )
                warnings.warn(warn_msg)
                self._last_model_warning = warn_msg
                self.model_name = fallback_model
                self._model, self._processor, self._tokenizer = self._load_clip_runtime(fallback_model)
                self._model_type = "clip"
                return self._model, self._processor, self._tokenizer, self._model_type
        elif model_type == "llava-onevision":
            self._model = AutoModel.from_pretrained(resolved_model_id, trust_remote_code=True, **token_kwargs)
            self._processor = AutoProcessor.from_pretrained(resolved_model_id, trust_remote_code=True, **token_kwargs)
            self._tokenizer = AutoTokenizer.from_pretrained(resolved_model_id, trust_remote_code=True, **token_kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        self._model.eval()
        self._model_type = model_type
        return self._model, self._processor, self._tokenizer, self._model_type

    @staticmethod
    def _device_for_model(model: Any) -> torch.device:
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @staticmethod
    def _mean_pool(hidden_state: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if attention_mask is None:
            return hidden_state.mean(dim=1)
        mask = attention_mask.unsqueeze(-1).to(hidden_state.dtype)
        masked = hidden_state * mask
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return masked.sum(dim=1) / denom

    def _prepare_inputs(self, inputs: dict[str, Any], device: torch.device) -> dict[str, Any]:
        prepared: dict[str, Any] = {}
        for key, value in inputs.items():
            if hasattr(value, "to"):
                prepared[key] = value.to(device)
            else:
                prepared[key] = value
        return prepared

    def _extract_features(self, outputs: Any, fallback_hidden_state: Optional[torch.Tensor] = None) -> np.ndarray:
        if hasattr(outputs, "text_embeds") and outputs.text_embeds is not None:
            vec = outputs.text_embeds
        elif hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
            vec = outputs.image_embeds
        elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            vec = outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            vec = outputs.last_hidden_state.mean(dim=1)
        elif fallback_hidden_state is not None:
            vec = fallback_hidden_state.mean(dim=1)
        else:
            raise RuntimeError("Could not extract embeddings from model outputs.")
        return vec.detach().cpu().numpy().astype("float32")

    def _forward_multimodal(self, inputs: dict[str, Any], model: Any) -> Any:
        try:
            return model(**inputs, output_hidden_states=True, return_dict=True)
        except TypeError:
            return model(**inputs, return_dict=True)
        except Exception:
            return model(**inputs)

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-12, a_max=None)
        return vectors / norms

    @staticmethod
    def _clean_text(raw: str) -> str:
        return raw.replace("_", " ").replace("-", " ").strip()

    @staticmethod
    def _normalize_key(raw: str) -> str:
        x = raw.lower().replace("_", " ").replace("-", " ")
        x = re.sub(r"[^a-z0-9\s]", " ", x)
        x = re.sub(r"\s+", " ", x).strip()
        return x

    @staticmethod
    def _tokenize_for_sparse(raw: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", raw.lower())

    def _record_to_sparse_doc(self, rec: dict[str, Any]) -> str:
        return " ".join(
            [
                str(rec.get("label", "")),
                str(rec.get("image_name", "")),
                str(rec.get("text", "")),
            ]
        ).strip()

    def _build_sparse_index(self) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise RuntimeError(
                "BM25 sparse retrieval requires rank-bm25. Install with: pip install rank-bm25"
            ) from exc

        tokenized_docs: list[list[str]] = []
        doc_ids: list[str] = []
        for rec in self.records:
            tokens = self._tokenize_for_sparse(self._record_to_sparse_doc(rec))
            if not tokens:
                tokens = ["pokemon"]
            tokenized_docs.append(tokens)
            doc_ids.append(rec["id"])

        self._bm25 = BM25Okapi(tokenized_docs)
        self._bm25_doc_ids = doc_ids

    @staticmethod
    def _rrf_fuse(
        ranked_ids: list[list[str]],
        weights: list[float],
        rrf_k: int,
    ) -> dict[str, float]:
        scores: dict[str, float] = {}
        for ids, weight in zip(ranked_ids, weights):
            if weight <= 0:
                continue
            for rank, item_id in enumerate(ids, start=1):
                scores[item_id] = scores.get(item_id, 0.0) + (weight / float(rrf_k + rank))
        return scores

    def _sparse_search(self, query_text: str, n_results: int) -> list[tuple[str, float]]:
        if self._bm25 is None:
            self._build_sparse_index()

        assert self._bm25 is not None
        tokens = self._tokenize_for_sparse(query_text)
        if not tokens:
            return []

        scores = np.asarray(self._bm25.get_scores(tokens), dtype="float32")
        if scores.size == 0:
            return []

        top_n = int(max(1, min(n_results, scores.size)))
        idx = np.argpartition(scores, -top_n)[-top_n:]
        idx = idx[np.argsort(scores[idx])[::-1]]

        max_score = float(np.max(scores[idx])) if idx.size else 0.0
        norm = max(max_score, 1e-8)

        out: list[tuple[str, float]] = []
        for i in idx:
            raw_score = float(scores[i])
            if raw_score <= 0:
                continue
            out.append((self._bm25_doc_ids[int(i)], raw_score / norm))
        return out

    @staticmethod
    def _token_overlap_score(a: str, b: str) -> float:
        a_tokens = set(a.split())
        b_tokens = set(b.split())
        if not a_tokens or not b_tokens:
            return 0.0
        inter = len(a_tokens.intersection(b_tokens))
        union = len(a_tokens.union(b_tokens))
        return inter / union

    def _load_cure_map(self, cure_json_path: Optional[str | Path]) -> dict[str, str]:
        if not cure_json_path:
            return {}

        path = Path(cure_json_path)
        if not path.is_absolute():
            path = (self.root / path).resolve()

        if not path.exists():
            raise FileNotFoundError(f"Cure json path does not exist: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw_map = json.load(f)

        cure_map: dict[str, str] = {}
        for k, v in raw_map.items():
            norm = self._normalize_key(str(k))
            if norm:
                cure_map[norm] = str(v).strip()
        return cure_map

    def _match_cure_text(self, label: str, cure_map: dict[str, str]) -> str:
        if not cure_map:
            return ""

        label_norm = self._normalize_key(label)
        if label_norm in cure_map:
            return cure_map[label_norm]

        close = get_close_matches(label_norm, cure_map.keys(), n=1, cutoff=0.86)
        if close:
            return cure_map[close[0]]

        best_key = ""
        best_score = 0.0
        for k in cure_map.keys():
            s = self._token_overlap_score(label_norm, k)
            if s > best_score:
                best_key = k
                best_score = s

        if best_key and best_score >= 0.5:
            return cure_map[best_key]

        return ""

    def discover_records(
        self,
        dataset_root: str | Path,
        cure_map: Optional[dict[str, str]] = None,
        max_images_per_class: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        dataset_root = Path(dataset_root).resolve()
        if not dataset_root.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_root}")

        cure_map = cure_map or {}
        records: list[dict[str, Any]] = []
        per_class_counts: dict[str, int] = {}

        for file in sorted(dataset_root.rglob("*")):
            if not file.is_file() or file.suffix.lower() not in IMAGE_EXTS:
                continue

            class_key = str(file.parent.relative_to(dataset_root))
            if max_images_per_class is not None:
                current_count = per_class_counts.get(class_key, 0)
                if current_count >= max_images_per_class:
                    continue

            parent_label = self._clean_text(file.parent.name)
            stem_name = self._clean_text(file.stem)
            try:
                stored_path = str(file.resolve().relative_to(self.root))
            except ValueError:
                stored_path = str(file.resolve())

            cure_text = self._match_cure_text(parent_label, cure_map)

            # Extract first 200 chars of cure as symptom description for better text matching
            symptom_desc = cure_text[:250] if cure_text else ""
            
            if cure_text:
                text_description = (
                    f"{parent_label}. "
                    f"Description: {symptom_desc}. "
                    f"Additional info: {cure_text}"
                )
            else:
                text_description = (
                    f"{parent_label}. "
                    "No extra descriptive information is available for this label."
                )

            records.append(
                {
                    "id": f"img_{len(records)}",
                    "image_path": stored_path,
                    "label": parent_label,
                    "image_name": stem_name,
                    "text": text_description,
                    "cure": cure_text,
                }
            )
            per_class_counts[class_key] = per_class_counts.get(class_key, 0) + 1

        if not records:
            raise RuntimeError(
                f"No images found under dataset path: {dataset_root}. "
                "Unzip the Kaggle dataset first."
            )

        return records


    def _embed_texts(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        model, processor, tokenizer, model_type = self._lazy_model()
        all_embeds: list[np.ndarray] = []

        effective_batch_size = batch_size
        if model_type == "janus":
            # Janus text forward is much heavier than CLIP; keep batches minimal.
            effective_batch_size = 1

        for start in range(0, len(texts), effective_batch_size):
            batch = texts[start : start + effective_batch_size]
            if model_type == "clip":
                inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True)
                with np.errstate(all="ignore"):
                    text_outputs = model.text_model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask"),
                        return_dict=True,
                    )
                    text_features = model.text_projection(text_outputs.pooler_output).detach().cpu().numpy().astype("float32")
                all_embeds.append(text_features)
            elif model_type == "janus":
                active_tokenizer = tokenizer or getattr(processor, "tokenizer", None)
                if active_tokenizer is None:
                    raise RuntimeError("Janus tokenizer is unavailable for text embedding.")

                inputs = active_tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
                device = self._device_for_model(model)
                inputs = self._prepare_inputs(inputs, device)

                language_model = getattr(model, "language_model", None)
                if language_model is None:
                    raise RuntimeError("Janus model does not expose language_model for text embedding.")

                lm_backbone = getattr(language_model, "model", language_model)
                with torch.no_grad():
                    outputs = lm_backbone(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask"),
                        output_hidden_states=False,
                        use_cache=False,
                        return_dict=True,
                    )

                hidden_state = getattr(outputs, "last_hidden_state", None)
                if hidden_state is None and hasattr(outputs, "hidden_states") and outputs.hidden_states:
                    hidden_state = outputs.hidden_states[-1]
                if hidden_state is None:
                    raise RuntimeError("Could not extract Janus text hidden states.")

                pooled = self._mean_pool(hidden_state, inputs.get("attention_mask"))
                all_embeds.append(pooled.detach().cpu().numpy().astype("float32"))

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                if tokenizer is not None:
                    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
                else:
                    inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True)
                device = self._device_for_model(model)
                inputs = self._prepare_inputs(inputs, device)
                outputs = self._forward_multimodal(inputs, model)
                hidden_state = getattr(outputs, "last_hidden_state", None)
                if hidden_state is None and hasattr(outputs, "hidden_states") and outputs.hidden_states:
                    hidden_state = outputs.hidden_states[-1]
                features = self._extract_features(outputs, fallback_hidden_state=hidden_state)
                all_embeds.append(features)

        return self._normalize(np.vstack(all_embeds))


    def _embed_images(self, image_paths: list[str], batch_size: int = 32) -> np.ndarray:
        model, processor, tokenizer, model_type = self._lazy_model()
        all_embeds: list[np.ndarray] = []

        effective_batch_size = batch_size
        if model_type == "janus":
            # Janus vision path is memory intensive during bulk indexing.
            effective_batch_size = 1

        for start in range(0, len(image_paths), effective_batch_size):
            batch_paths = image_paths[start : start + effective_batch_size]
            images = []
            for p in batch_paths:
                raw = Path(p)
                full = raw if raw.is_absolute() else (self.root / raw)
                with Image.open(full) as img:
                    images.append(img.convert("RGB"))
            if model_type == "clip":
                inputs = processor(images=images, return_tensors="pt")
                with np.errstate(all="ignore"):
                    vision_outputs = model.vision_model(
                        pixel_values=inputs["pixel_values"],
                        return_dict=True,
                    )
                    img_features = model.visual_projection(vision_outputs.pooler_output).detach().cpu().numpy().astype("float32")
                all_embeds.append(img_features)
            elif model_type == "janus":
                image_processor = getattr(processor, "image_processor", None)
                if image_processor is None:
                    raise RuntimeError("Janus processor does not expose image_processor.")

                prepared = image_processor(images, return_tensors="pt")
                if isinstance(prepared, dict):
                    pixel_values = prepared["pixel_values"]
                else:
                    pixel_values = prepared.pixel_values

                device = self._device_for_model(model)
                pixel_values = pixel_values.to(device)

                with torch.no_grad():
                    vision_outputs = model.vision_model(pixel_values)

                if isinstance(vision_outputs, tuple):
                    vision_hidden = vision_outputs[0]
                elif hasattr(vision_outputs, "last_hidden_state"):
                    vision_hidden = vision_outputs.last_hidden_state
                else:
                    vision_hidden = vision_outputs

                if hasattr(model, "aligner"):
                    vision_hidden = model.aligner(vision_hidden)

                if vision_hidden.ndim == 3:
                    pooled = vision_hidden.mean(dim=1)
                elif vision_hidden.ndim == 2:
                    pooled = vision_hidden
                else:
                    raise RuntimeError(f"Unexpected Janus vision tensor shape: {tuple(vision_hidden.shape)}")

                all_embeds.append(pooled.detach().cpu().numpy().astype("float32"))

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                if hasattr(processor, "__call__"):
                    inputs = processor(images=images, return_tensors="pt")
                else:
                    raise RuntimeError(f"Processor for {self.model_name} cannot handle images.")

                device = self._device_for_model(model)
                inputs = self._prepare_inputs(inputs, device)
                outputs = self._forward_multimodal(inputs, model)
                hidden_state = getattr(outputs, "last_hidden_state", None)
                if hidden_state is None and hasattr(outputs, "hidden_states") and outputs.hidden_states:
                    hidden_state = outputs.hidden_states[-1]
                features = self._extract_features(outputs, fallback_hidden_state=hidden_state)
                all_embeds.append(features)

        return self._normalize(np.vstack(all_embeds))

    def build_index(
        self,
        dataset_root: str | Path,
        overwrite: bool = False,
        cure_json_path: Optional[str | Path] = None,
        max_images_per_class: Optional[int] = 10,
    ) -> None:
        if self.metadata_path.exists() and not overwrite:
            raise FileExistsError(
                f"Vector DB already exists at {self.vector_dir}. Use overwrite=True to rebuild."
            )

        if overwrite and self.vector_dir.exists():
            shutil.rmtree(self.vector_dir)

        self.vector_dir.mkdir(parents=True, exist_ok=True)
        cure_map = self._load_cure_map(cure_json_path)
        self.records = self.discover_records(
            dataset_root,
            cure_map=cure_map,
            max_images_per_class=max_images_per_class,
        )
        self._record_by_id = {x["id"]: x for x in self.records}

        self._connect_collections()
        assert self.text_collection is not None
        assert self.image_collection is not None

        texts = [x["text"] for x in self.records]
        image_paths = [x["image_path"] for x in self.records]
        ids = [x["id"] for x in self.records]
        metadatas = [
            {"label": x["label"], "image_name": x["image_name"], "image_path": x["image_path"]}
            for x in self.records
        ]

        text_emb = self._embed_texts(texts)
        image_emb = self._embed_images(image_paths)

        batch_size = 512
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            batch_ids = ids[start:end]
            batch_meta = metadatas[start:end]
            batch_docs = texts[start:end]

            self.text_collection.add(
                ids=batch_ids,
                embeddings=text_emb[start:end].tolist(),
                metadatas=batch_meta,
                documents=batch_docs,
            )

            self.image_collection.add(
                ids=batch_ids,
                embeddings=image_emb[start:end].tolist(),
                metadatas=batch_meta,
            )

        metadata = {
            "model_name": self.model_name,
            "record_count": len(self.records),
            "dimension": int(text_emb.shape[1]),
            "cure_entries": len(cure_map),
            "records_with_cure": sum(1 for x in self.records if x.get("cure")),
            "records": self.records,
            "collections": {
                "text": self.text_collection_name,
                "image": self.image_collection_name,
            },
        }
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        self._build_sparse_index()

    def load_index(self) -> None:
        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Missing vector DB files under {self.vector_dir}. Run build_index first."
            )

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        self.records = payload["records"]
        self._record_by_id = {x["id"]: x for x in self.records}
        self.text_collection_name = payload.get("collections", {}).get("text", self.text_collection_name)
        self.image_collection_name = payload.get("collections", {}).get("image", self.image_collection_name)
        self._connect_collections()
        self._build_sparse_index()


    def embed_query_text(self, query: str) -> np.ndarray:
        return self._embed_texts([query])


    def embed_query_image(self, image_path: str | Path) -> np.ndarray:
        return self._embed_images([image_path])

    def _mode_from_auto(self, config: RetrievalConfig, query_image_path: Optional[str]) -> str:
        if config.mode != "auto":
            return config.mode
        return "hybrid" if query_image_path else "text_only"

    def search(
        self,
        query_text: str,
        query_image_path: Optional[str],
        config: RetrievalConfig,
        sparse_query: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        if self.text_collection is None or self.image_collection is None:
            self.load_index()

        assert self.text_collection is not None
        assert self.image_collection is not None

        mode = self._mode_from_auto(config, query_image_path)

        text_vec = self.embed_query_text(query_text)
        image_vec = self.embed_query_image(query_image_path) if query_image_path else None

        if image_vec is not None:
            joint = self._normalize(config.text_weight * text_vec + config.image_weight * image_vec)
        else:
            joint = text_vec

        dense_map: dict[str, dict[str, Any]] = {}
        dense_candidate_k = max(config.top_k, int(config.dense_candidate_k))
        sparse_candidate_k = max(config.top_k, int(config.sparse_candidate_k))

        def score_from_distance(distance: float) -> float:
            # With cosine space, Chroma returns distance ~= (1 - cosine_similarity).
            return 1.0 - float(distance)

        def run_dense_stage1() -> dict[str, dict[str, Any]]:
            dense_local: dict[str, dict[str, Any]] = {}

            if mode in {"text_only", "hybrid"}:
                text_res = self.text_collection.query(
                    query_embeddings=joint.tolist(),
                    n_results=dense_candidate_k,
                    include=["distances"],
                )
                for item_id, dist in zip(text_res["ids"][0], text_res["distances"][0]):
                    score = score_from_distance(dist)
                    if score < config.text_threshold:
                        continue
                    rec = self._record_by_id[item_id]
                    dense_local[item_id] = {
                        "id": item_id,
                        "text_score": float(score),
                        "image_score": 0.0,
                        "dense_score": 0.0,
                        "sparse_score": 0.0,
                        "label": rec["label"],
                    }

            if mode in {"image_only", "hybrid"}:
                image_query = image_vec if mode == "image_only" and image_vec is not None else joint
                image_res = self.image_collection.query(
                    query_embeddings=image_query.tolist(),
                    n_results=dense_candidate_k,
                    include=["distances"],
                )
                for item_id, dist in zip(image_res["ids"][0], image_res["distances"][0]):
                    score = score_from_distance(dist)
                    if score < config.image_threshold:
                        continue
                    if item_id not in dense_local:
                        rec = self._record_by_id[item_id]
                        dense_local[item_id] = {
                            "id": item_id,
                            "text_score": 0.0,
                            "image_score": 0.0,
                            "dense_score": 0.0,
                            "sparse_score": 0.0,
                            "label": rec["label"],
                        }
                    dense_local[item_id]["image_score"] = float(score)

            for payload in dense_local.values():
                payload["dense_score"] = float(
                    config.text_weight * payload["text_score"] + config.image_weight * payload["image_score"]
                )
            return dense_local

        use_sparse = mode in {"text_only", "hybrid"}

        def run_sparse_stage1() -> list[tuple[str, float]]:
            if not use_sparse:
                return []
            sparse_query_text = (sparse_query or query_text or "").strip()
            return self._sparse_search(sparse_query_text, n_results=sparse_candidate_k)

        # Stage 1: run dense ANN and sparse BM25 retrieval in parallel.
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                dense_future = pool.submit(run_dense_stage1)
                sparse_future = pool.submit(run_sparse_stage1)
                dense_map = dense_future.result()
                sparse_hits = sparse_future.result()
        except Exception:
            # Fallback for environments where backend clients dislike threaded queries.
            dense_map = run_dense_stage1()
            sparse_hits = run_sparse_stage1()

        dense_ranked_ids = [
            item_id
            for item_id, _ in sorted(
                ((item_id, payload["dense_score"]) for item_id, payload in dense_map.items()),
                key=lambda x: x[1],
                reverse=True,
            )
        ]

        sparse_ranked_ids: list[str] = []
        sparse_scores: dict[str, float] = {}
        if use_sparse:
            sparse_ranked_ids = [item_id for item_id, _ in sparse_hits]
            sparse_scores = {item_id: score for item_id, score in sparse_hits}

        rrf_scores = self._rrf_fuse(
            ranked_ids=[dense_ranked_ids, sparse_ranked_ids],
            weights=[config.dense_rrf_weight, config.sparse_rrf_weight],
            rrf_k=max(1, int(config.rrf_k)),
        )

        candidate_ids = set(dense_ranked_ids).union(sparse_ranked_ids)

        results: list[dict[str, Any]] = []
        for item_id in candidate_ids:
            payload = dense_map.get(
                item_id,
                {
                    "text_score": 0.0,
                    "image_score": 0.0,
                    "dense_score": 0.0,
                },
            )
            rec = self._record_by_id[item_id]
            sparse_score = float(sparse_scores.get(item_id, 0.0))
            fusion = float(rrf_scores.get(item_id, 0.0))
            results.append(
                {
                    "id": rec["id"],
                    "image_path": rec["image_path"],
                    "label": rec["label"],
                    "image_name": rec["image_name"],
                    "text": rec["text"],
                    "text_score": payload["text_score"],
                    "image_score": payload["image_score"],
                    "dense_score": float(payload["dense_score"]),
                    "sparse_score": sparse_score,
                    "rrf_score": fusion,
                    "fusion_score": float(fusion),
                }
            )

        results.sort(key=lambda x: x["fusion_score"], reverse=True)
        final = results[: config.top_k]
        for i, item in enumerate(final, start=1):
            item["citation_id"] = f"R{i}"
        return final


MultimodalFaissStore = MultimodalChromaStore


def ensure_relative_to_root(root: str | Path, target: str | Path) -> str:
    root = Path(root).resolve()
    target = Path(target).resolve()
    try:
        return str(target.relative_to(root))
    except ValueError:
        return str(target)
