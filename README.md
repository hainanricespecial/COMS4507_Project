# Plant Multimodal Agent (INFS4205 A3)

## Updated Architecture

Current end-to-end flow:

1. User query (+ optional image)
2. Query rewrite and sparse summary generation (keyword-focused)
3. Retrieval routing decision:
  - retrieve fresh evidence from index, or
  - answer from in-chat memory when follow-up can be resolved without new retrieval
4. Dual retrieval when retrieval is selected:
  - Dense retrieval: CLIP-based vector retrieval from Chroma
  - Sparse retrieval: BM25 lexical retrieval over record text/labels
5. Rank fusion and reranking with Reciprocal Rank Fusion (RRF)
6. Grounded generation (Ollama LLM) or heuristic fallback

Conversation memory is preserved per chat session and fed into rewrite/routing/answer steps.

A chat-based multimodal RAG system for plant disease reasoning, built with:

- CLIP (`openai/clip-vit-base-patch16`) for text/image embeddings
- BM25 sparse retriever (`rank-bm25`) for lexical matching
- Chroma vector database (separate text and image collections)
- LangGraph workflow for retrieval -> answer orchestration
- Streamlit chat interface with image upload

This structure is intentionally similar to `LangGraph_Agent/`, but redesigned for your personalised multimodal assignment requirements and ablation studies.

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         User Input (Streamlit App)                          │
│                          - Text query                                        │
│                          - Optional image                                    │
└────────────────────────┬────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Query Processing (local_agent.py)                        │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ 1. Query Rewrite Node (LLM or Heuristic)                            │  │
│  │    - Expand keywords (e.g. "red spot" → "leaf discoloration blight")│  │
│  │    - Extract disease/symptom/treatment terms                        │  │
│  │    - Input: user_query, chat_history                               │  │
│  │    - Output: rewritten_query, summarized_query                      │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                         │                                                    │
│                         ▼                                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ 2. Routing Decision Node (LLM or Heuristic)                         │  │
│  │    Rules:                                                            │  │
│  │    - Image present? → RETRIEVE                                       │  │
│  │    - Referential pronoun ("it", "this")? → RETRIEVE                 │  │
│  │    - Follow-up question? → check memory first                       │  │
│  │    - Intent keywords (symptom, diagnosis, treatment)? → RETRIEVE    │  │
│  │    - Input: rewritten_query, memory_context                         │  │
│  │    - Output: use_retrieval (True/False), decision_reason            │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────┬──────────────────────────────────────────────────┘
                             │
                    ┌────────┴────────┐
                    │                 │
         (RETRIEVE) │                 │ (MEMORY)
                    ▼                 ▼
     ┌──────────────────────┐  ┌─────────────────┐
     │ Dual Retrieval Flow  │  │ Answer from     │
     │ (vector_store.py)    │  │ Chat History    │
     │                      │  │                 │
     │ ┌──────────────────┐ │  │ Format:         │
     │ │ Dense Retrieval  │ │  │ - Last N turns  │
     │ │ via CLIP+Chroma  │ │  │ - Memory context│
     │ │                  │ │  │ - Grounded LLM  │
     │ │ Returns: 40 docs │ │  └─────────────────┘
     │ │ Score: cosine    │ │           │
     │ │ similarity       │ │           │
     │ └────────┬─────────┘ │           │
     │          │           │           │
     │ ┌────────▼─────────┐ │           │
     │ │ Sparse Retrieval │ │           │
     │ │ via BM25 (rank)  │ │           │
     │ │                  │ │           │
     │ │ Returns: 40 docs │ │           │
     │ │ Score: BM25      │ │           │
     │ │ lexical match    │ │           │
     │ └────────┬─────────┘ │           │
     │          │           │           │
     │ ┌────────▼─────────────────────┐│
     │ │  RRF Fusion (Rank Fusion)    ││
     │ │                              ││
     │ │ fused_score =                ││
     │ │   dense_weight / (k + rank1) ││
     │ │ + sparse_weight / (k + rank2)││
     │ │ (k=60, default k=1.0 each)   ││
     │ │                              ││
     │ │ Returns: top-5 fused results ││
     │ └────────┬─────────────────────┘│
     │          │                       │
     └──────────┼───────────────────────┘
                │
                ▼
     ┌──────────────────────┐
     │  Generation          │
     │  (Ollama LLM)        │
     │                      │
     │  Inputs:             │
     │  - Retrieved items   │
     │  - Rewritten query   │
     │  - Chat memory       │
     │  - All relevance     │
     │    scores shown      │
     │                      │
     │  Output:             │
     │  - Answer            │
     │  - Cure/treatment    │
     │  - Confidence score  │
     └──────────┬───────────┘
                │
                ▼
     ┌──────────────────────┐
     │  User Response       │
     │  (Streamlit UI)      │
     │                      │
     │  - Answer text       │
     │  - Scoring details   │
     │  - Rewrite info      │
     │  - Routing reason    │
     │  - Cure plan (if any)│
     └──────────────────────┘
```

## Module Interaction Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Offline: Indexing Phase                          │
│                                                                         │
│  build_index.py                                                         │
│  ├─→ Load dataset + cure.json                                           │
│  ├─→ vector_store.build_index()                                         │
│  │   ├─→ Embed texts with CLIP                                          │
│  │   ├─→ Embed images with CLIP                                         │
│  │   ├─→ Store in Chroma (2 collections: text, image)                  │
│  │   └─→ _build_sparse_index()  [BM25 built from records]              │
│  │       ├─ Tokenize: label + image_name + text                        │
│  │       └─ BM25Okapi(tokenized_docs)                                   │
│  └─→ Save: vector_db/chroma/metadata.json + collections                │
│      (BM25 index rebuilt on-load from stored records)                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                      Online: Chat/Query Phase                           │
│                                                                         │
│  app.py (Streamlit UI)                                                 │
│  └─→ st.session_state.messages (in-chat memory)                        │
│  └─→ load_agent() creates PlantMultimodalAgent                         │
│      │                                                                 │
│      └─→ local_agent.PlantMultimodalAgent                              │
│          ├─ __init__: load vector_store + build LangGraph             │
│          │   │                                                        │
│          │   └─→ vector_store.py::load_index()                        │
│          │       ├─ Load metadata.json + records                      │
│          │       ├─ Connect to Chroma collections                     │
│          │       └─ _build_sparse_index() [BM25 from records]         │
│          │                                                            │
│          └─ invoke(query, query_image_path, config, chat_history)    │
│              │                                                        │
│              ├─→ Rewrite Node: _rewrite_node()                        │
│              │   └─→ Uses Ollama (or heuristic) + _tokenize_for_sparse│
│              │                                                        │
│              ├─→ Decide Node: _decide_node()                          │
│              │   └─→ Uses Ollama (or heuristic) rules                 │
│              │                                                        │
│              ├─→ Retrieve Node (conditional): _retrieve_node()        │
│              │   └─→ vector_store.search()  [DUAL RETRIEVAL]         │
│              │       ├─ Dense search: text_collection + image_collection
│              │       ├─ Sparse search: self._sparse_search(BM25)      │
│              │       └─ Fuse: _rrf_fuse(dense_results, sparse_results)
│              │                                                        │
│              └─→ Answer Node: _make_grounded_prompt()                │
│                  └─→ Ollama LLM with grounded context                │
│                  └─→ Response + all scores displayed                 │
│                                                                     │
│  eval_ablation.py                                                      │
│  └─→ Ablation experiments: text_only, image_only, hybrid modes        │
│      └─→ Metrics: Recall@K, latency                                  │
│                                                                     │
│  run_full_evaluation_suite.py                                          │
│  └─→ Comprehensive evaluation with bootstrap CI                       │
│      └─→ Metrics: BLEU, ROUGE, BERTScore, CLIPScore                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Key Components Reference

| File | Purpose | Key Methods |
|------|---------|-------------|
| `vector_store.py` | Dense + sparse indexing & retrieval | `build_index()`, `load_index()`, `search()`, `_build_sparse_index()`, `_sparse_search()`, `_rrf_fuse()` |
| `local_agent.py` | LangGraph workflow orchestration | `invoke()`, `_rewrite_node()`, `_decide_node()`, `_retrieve_node()`, `_make_grounded_prompt()` |
| `app.py` | Streamlit web UI with chat memory | Chat interface, session state, load_agent() |
| `build_index.py` | Offline indexing pipeline | Command-line dataset → index builder |
| `eval_ablation.py` | Retrieval quality evaluation | Benchmark runner, Recall@K metrics |
| `run_full_evaluation_suite.py` | Full evaluation with generation | Multi-model ablation with BLEU/ROUGE/BERTScore |

## Folder Structure

- `build_index.py`: build text/image Chroma collections from your dataset
- `vector_store.py`: CLIP embedding + indexing + retrieval core
- `local_agent.py`: LangGraph-based multimodal agent
- `app.py`: chat interface (text + image input)
- `eval_ablation.py`: quantitative retrieval ablation runner
- `benchmark_queries.json`: benchmark template covering required query families

## 1) Prepare Dataset

Download and unzip the Kaggle dataset:

- https://www.kaggle.com/datasets/sadmansakibmahi/plant-disease-expert

Place the extracted folder anywhere local (example path below).

## 2) Install Dependencies

```bash
cd Plant_Multimodal_Agent
pip install -r requirements.txt
```

## 3) Build Vector Database

```bash
python build_index.py \
  --root . \
  --dataset-root "../17/Image Data base/Image Data base" \
  --cure-json "../17/plant diseases cure/cure.json" \
  --overwrite
```

If your paths differ, override them with absolute paths.

This creates:

- `vector_db/chroma/` persistence directory
- `vector_db/chroma/metadata.json`

### Build Two CLIP Indexes For Comparison

Yes: `openai/clip-vit-base-patch16` and `openai/clip-vit-base-patch32` both produce 512-d embeddings.
But use separate indexes for fair comparison, because they are different embedding spaces.

Build CLIP patch16 index:

```bash
python build_index.py \
  --root . \
  --dataset-root "../17/Image Data base/Image Data base" \
  --cure-json "../17/plant diseases cure/cure.json" \
  --model-name openai/clip-vit-base-patch16 \
  --vector-subdir vector_db/chroma_clip16 \
  --overwrite
```

Build CLIP patch32 index:

```bash
python build_index.py \
  --root . \
  --dataset-root "../17/Image Data base/Image Data base" \
  --cure-json "../17/plant diseases cure/cure.json" \
  --model-name openai/clip-vit-base-patch32 \
  --vector-subdir vector_db/chroma_clip32 \
  --overwrite
```

Run app with CLIP16 index:

```bash
HF_MODEL_NAME=openai/clip-vit-base-patch16 PMA_VECTOR_SUBDIR=vector_db/chroma_clip16 streamlit run app.py
```

Run app with CLIP32 index:

```bash
HF_MODEL_NAME=openai/clip-vit-base-patch32 PMA_VECTOR_SUBDIR=vector_db/chroma_clip32 streamlit run app.py
```


## Important: Ollama Server Requirement

If you are using any Ollama models (for query rewriting, routing, or generation), you **must** start the Ollama server before running the app or any scripts that use Ollama. Otherwise, model calls will fail.

Start the Ollama server in a separate terminal:

```bash
ollama serve
```

Leave this running in the background while you use the chat interface, CLI, or evaluation scripts.

---

## 4) Run Chat Interface

```bash
streamlit run app.py
```

Features in the UI:

- chat input + optional image upload
- retrieval mode switch: `text_only`, `image_only`, `hybrid`, `auto`
- configurable top-k
- configurable text/image score weights
- threshold controls for retrieval filtering
- optional Ollama generation
- retrieval can surface cure/action-plan text when available in the cure JSON

## 5) CLI Agent (No UI)

```bash
python local_agent.py \
  --root . \
  --query "what happen to this plant?" \
  --query-image /absolute/path/to/query_leaf.jpg \
  --mode hybrid
```

## 6) Ablation / Evaluation

Edit `benchmark_queries.json` with your own cases and expected labels.

Run one variant:

```bash
python eval_ablation.py --root . --mode text_only --top-k 5
python eval_ablation.py --root . --mode image_only --top-k 5
python eval_ablation.py --root . --mode hybrid --top-k 5 --text-weight 0.4 --image-weight 0.6
```

Suggested metrics reported:

- average Recall@K (label hit)
- average retrieval latency (ms)

### Full Evaluation Suite (Retrieval + Generation + Grounding)

Run the all-in-one script to evaluate:

- retrieval quality across families
- generation quality (BLEU / ROUGE / BERTScore)
- multimodal grounding quality (CLIPScore)
- uncertainty bounds via bootstrap confidence intervals
- ablation pipelines: generation-only (no index), retrieval-only, fixed RAG, routed RAG, memory-aware RAG
- toggle ablations on RAG behavior:
  - with/without query rewrite
  - with/without sparse summary
  - with/without retrieval
  - with/without memory
  - with/without routing policy
- ablations over top-k and hybrid text/image weights

Default benchmark file:

- `benchmark_queries_extended.json`

Default model matrix:

- embeddings: `openai/clip-vit-base-patch16`, `openai/clip-vit-base-patch32`
- generation: `llava:7b`, `llava-phi3:3.8b`

Run once:

```bash
python run_full_evaluation_suite.py \
  --root . \
  --benchmark benchmark_queries_extended.json \
  --embedding-models openai/clip-vit-base-patch16,openai/clip-vit-base-patch32 \
  --vector-subdirs vector_db/chroma_clip16,vector_db/chroma_clip32 \
  --generation-models llava:7b,llava-phi3:3.8b \
  --top-k-values 3,5 \
  --weight-pairs 0.7:0.3,0.5:0.5,0.3:0.7 \
  --parallel-jobs 4 \
  --bootstrap-samples 1000 \
  --ci-level 0.95 \
  --bootstrap-seed 42 \
  --save-dir evaluation_results/full_suite
```

Disable toggle ablations (optional):

```bash
python run_full_evaluation_suite.py --disable-toggle-ablations
```

Tips:

- increase `--parallel-jobs` to run embedding x generation pairs concurrently (set this based on CPU/RAM/Ollama capacity)
- reduce `--bootstrap-samples` (for example to `300`) for faster exploratory runs

Main outputs:

- `evaluation_results/full_suite/summary_all_runs.json`
- `evaluation_results/full_suite/leaderboard.csv`
- `evaluation_results/full_suite/question_answers.json`
- `evaluation_results/full_suite/question_answers.md`

Per-run detailed files are also saved under `evaluation_results/full_suite/`.

## 7) Run API Version (FastAPI)

This project now also includes an API wrapper similar in spirit to your reference repo style: reusable app object + external HTTP access.

Start the API server:

```bash
uvicorn api_app:app --host 0.0.0.0 --port 8000
```

Optional environment variables:

- `PMA_ROOT`: project root path (defaults to current folder where `api_app.py` lives)
- `PMA_API_HOST`: host for direct `python api_app.py` run
- `PMA_API_PORT`: port for direct `python api_app.py` run

Open Swagger UI:

- `http://127.0.0.1:8000/docs`

### JSON endpoint

`POST /query`

```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "what disease is this tomato leaf?",
    "query_image_path": "samples/Tomato Late blight/<your_image>.JPG",
    "retrieval": {
      "mode": "hybrid",
      "top_k": 5,
      "text_weight": 0.5,
      "image_weight": 0.5,
      "text_threshold": 0.05,
      "image_threshold": 0.10
    },
    "generation": {
      "use_llm": false,
      "model_name": "llava-phi3:3.8b",
      "temperature": 0.2
    }
  }'
```

### Multipart endpoint (upload image directly)

`POST /query-upload`

```bash
curl -X POST "http://127.0.0.1:8000/query-upload" \
  -F "query=what disease is this leaf" \
  -F "mode=hybrid" \
  -F "top_k=5" \
  -F "image=@/absolute/path/to/query_leaf.jpg"
```

### Utility endpoints

- `GET /health`: checks index load status
- `GET /models`: lists supported embedding models

## Notes for Your Report

- Personalisation point: your knowledge base is your curated plant dataset instance and your query suite.
- Originality point: compare retrieval designs and show failure analysis.
- Required comparisons: plain LLM vs agent and at least one ablation. You can disable generation in UI/agent to isolate retrieval quality.
