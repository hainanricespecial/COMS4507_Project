# Pokémon Multimodal Agent

This repository contains a multimodal Pokémon retrieval and question-answering project built for COMS4507. It combines CLIP-style image/text embeddings, ChromaDB retrieval, LangGraph-based reasoning, and an optional Ollama vision-language model for generation.

## Project Structure

- `app.py` - Streamlit UI for interactive chat and retrieval ablations.
- `local_agent.py` - Command-line entry point for a single query.
- `build_index.py` - Builds the Chroma vector index from the dataset.
- `eval_ablation.py` - Runs retrieval ablations on benchmark queries.
- `run_all_ablation.py` - Executes the default ablation sweep.
- `run_generation_model_ablation.py` - Compares embedding and generation model combinations.
- `run_full_evaluation_suite.py` - Runs the broader evaluation matrix.
- `vector_store.py` - Retrieval, embedding, and persistence logic.
- `4/` - Dataset files, including `pokemon.csv` and image assets.

## Requirements

- Python 3.11+ is recommended.
- Install dependencies with:

```bash
pip install -r requirements.txt
```

- If you want LLM-backed generation, install and start Ollama.
- The code reads environment variables from a `.env` file when present. Useful values include:
  - `HF_TOKEN` for Hugging Face model access.
  - `HF_MODEL_NAME` to change the default embedding model.
  - `PMA_VECTOR_SUBDIR` to change the default vector store location.

## Dataset Setup

The default dataset layout is already included under `4/`.

- Images are expected under `4/images/` or directly under `4/`.
- `4/pokemon.csv` is used to map images to labels and optional descriptions.

If you need to download the dataset again, the helper script is:

```bash
python kaggleHubDataset.py
```

## Build The Index

Build the vector database before running the app or evaluations:

```bash
python build_index.py --root . --dataset-root 4 --text-csv 4/pokemon.csv --vector-subdir vector_db/chroma --overwrite
```

Common variations:

- `--model-name openai/clip-vit-base-patch16` or `openai/clip-vit-base-patch32`
- `--vector-subdir vector_db/chroma_clip16` to keep multiple indexes side by side
- `--max-images-per-class 10` for faster test builds

## Run The App

Start the Streamlit interface after the index has been built:

```bash
streamlit run app.py
```

In the sidebar you can adjust retrieval mode, top-k, text/image weights, thresholds, and whether Ollama generation is enabled.

## Command-Line Query

Run a single query directly from the terminal:

```bash
python local_agent.py --root . --query "What Pokémon is this?" --query-image path/to/image.png --mode hybrid --top-k 5
```

Useful flags:

- `--use-llm` to enable Ollama-backed generation.
- `--rebuild` to rebuild the index before running the query.
- `--dataset-root 4` and `--text-csv 4/pokemon.csv` to override data paths.

## Evaluation

Run the retrieval ablation on the default benchmark:

```bash
python eval_ablation.py --root . --benchmark benchmark_queries.json --mode hybrid --top-k 5 --save-json ablation_results/hybrid.json
```

Run the default ablation sweep:

```bash
python run_all_ablation.py
```

Run the generation model comparison:

```bash
python run_generation_model_ablation.py
```

By default this writes to `ablation_results/generation_model_ablation/`.

Run the broader evaluation suite:

```bash
python run_full_evaluation_suite.py
```

By default this writes to `evaluation_results/full_suite/`.

## Notes

- The first retrieval may take longer while the embedding model loads.
- If NLTK stopwords are missing, the code downloads them automatically on first use.
- If you change the embedding model or vector-store path, rebuild the index so the store and model stay aligned.