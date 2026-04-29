from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

try:
    from .local_agent import GenerationConfig, PlantMultimodalAgent
    from .vector_store import MultimodalChromaStore, RetrievalConfig, ensure_relative_to_root
except ImportError:
    from COMS4507_Project.local_agent import GenerationConfig, PlantMultimodalAgent
    from COMS4507_Project.vector_store import MultimodalChromaStore, RetrievalConfig, ensure_relative_to_root


st.set_page_config(page_title="Plant Multimodal Agent", page_icon="🌿", layout="wide")


@st.cache_resource
def load_store(project_root: str) -> MultimodalChromaStore:
    store = MultimodalChromaStore(root=project_root)
    store.load_index()
    return store


@st.cache_resource
def load_agent(
    project_root: str,
    use_llm: bool,
    llm_model: str,
    enable_query_rewrite: bool,
    enable_query_summarize: bool,
    max_memory_turns: int,
) -> PlantMultimodalAgent:
    store = load_store(project_root)
    return PlantMultimodalAgent(
        store=store,
        generation=GenerationConfig(use_llm=use_llm, model_name=llm_model),
        enable_query_rewrite=enable_query_rewrite,
        enable_query_summarize=enable_query_summarize,
        max_memory_turns=max_memory_turns,
    )


def save_uploaded_image(upload) -> Path:
    suffix = Path(upload.name).suffix or ".jpg"
    tmp_dir = Path(tempfile.gettempdir())
    path = tmp_dir / f"query_upload_{next(tempfile._get_candidate_names())}{suffix}"
    path.write_bytes(upload.read())
    return path


def main() -> None:
    st.title("Plant Disease Multimodal RAG Agent")
    st.caption("Multimodal embeddings + Chroma + LangGraph with easy ablation controls")

    project_root = str(Path(__file__).resolve().parent)

    with st.sidebar:
        st.subheader("Ablation Controls")
        mode = st.selectbox("Retrieval mode", ["auto", "hybrid", "text_only", "image_only"], index=1)
        top_k = st.slider("Top-K", min_value=1, max_value=20, value=5)
        text_weight = st.slider("Text weight", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        image_weight = st.slider("Image weight", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        text_threshold = st.slider("Text threshold", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
        image_threshold = st.slider("Image threshold", min_value=0.0, max_value=1.0, value=0.10, step=0.01)

        st.subheader("Generation")
        use_llm = st.checkbox("Use Ollama VLM", value=False)
        llm_model = st.text_input("Ollama model", value="llava-phi3:3.8b")

        st.subheader("Query Processing")
        enable_query_rewrite = st.checkbox("Rewrite query before retrieval", value=True)
        enable_query_summarize = st.checkbox("Summarize query for BM25", value=True)
        max_memory_turns = st.slider("Memory turns kept", min_value=2, max_value=20, value=8)

    agent = load_agent(
        project_root,
        use_llm,
        llm_model,
        enable_query_rewrite,
        enable_query_summarize,
        max_memory_turns,
    )
    model_info = agent.store.get_embedding_model_info(ensure_loaded=False)

    with st.sidebar:
        st.divider()
        st.subheader("Runtime Model Status")
        st.write(f"Embedding requested: {model_info['requested_model']}")
        st.caption(
            f"Python: {model_info.get('python_version', 'unknown')} | "
            f"Transformers: {model_info.get('transformers_version', 'unknown')}"
        )
        if model_info["is_loaded"]:
            model_type = model_info["model_type"] or "unknown"
            st.write(f"Embedding active: {model_info['active_model']} ({model_type})")
        else:
            st.write("Embedding active: not loaded yet (loads on first retrieval)")

        if use_llm:
            st.write(f"Generation configured: {llm_model}")
            st.caption("Actual generation source is shown under each assistant response.")
        else:
            st.write("Generation configured: heuristic mode (Ollama disabled)")

        if model_info.get("warning"):
            st.warning(model_info["warning"])
        if model_info.get("error"):
            st.error(model_info["error"])

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("image"):
                st.image(msg["image"], caption="uploaded query image", width=280)

    query = st.chat_input("Ask about your plant disease dataset...")
    uploaded = st.file_uploader("Optional query image", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=False)

    # Add helpful guidance message
    if mode in ["text_only", "auto"] and not query:
        st.info(
            "💡 **Tip for best results:** Upload a leaf image alongside your text query. "
            "Image-based retrieval is much more accurate than text-only matching for disease diagnosis!"
        )
    elif mode == "text_only":
        st.warning(
            "⚠️ **Text-only mode**: For more accurate disease diagnosis, consider uploading a leaf image "
            "and switching to 'hybrid' or 'image_only' mode."
        )

    if query:
        img_path = None
        show_image = None
        if uploaded is not None:
            saved = save_uploaded_image(uploaded)
            img_path = ensure_relative_to_root(project_root, saved)
            show_image = str(saved)

        st.session_state.messages.append({"role": "user", "content": query, "image": show_image})

        with st.chat_message("user"):
            st.markdown(query)
            if show_image:
                st.image(show_image, caption="uploaded query image", width=280)

        cfg = RetrievalConfig(
            mode=mode,
            top_k=top_k,
            text_weight=text_weight,
            image_weight=image_weight,
            text_threshold=text_threshold,
            image_threshold=image_threshold,
        )

        with st.chat_message("assistant"):
            with st.spinner("Running retrieval + agent reasoning..."):
                out = agent.invoke(
                    query,
                    img_path,
                    cfg,
                    chat_history=st.session_state.messages,
                )
                current_model_info = agent.store.get_embedding_model_info(ensure_loaded=False)

            st.markdown(out["answer"])
            model_type = current_model_info["model_type"] or "unknown"
            generation_mode_used = out.get("generation_mode_used", "unknown")
            generation_model_used = out.get("generation_model_used", "unknown")
            st.caption(
                f"Embedding backend used: {current_model_info['active_model']} ({model_type}) | "
                f"Generation used: {generation_model_used} ({generation_mode_used})"
            )
            generation_note = out.get("generation_note", "")
            if generation_note:
                st.warning(generation_note)

            st.caption(
                f"Query rewrite: {out.get('rewritten_query', query)} | "
                f"Sparse summary: {out.get('summarized_query', query)}"
            )
            st.caption(
                f"Routing: {'retrieval' if out.get('use_retrieval', True) else 'memory'} | "
                f"Reason: {out.get('retrieval_decision_reason', '')}"
            )

            st.markdown("### Retrieved evidence")
            for i, item in enumerate(out["retrieved_items"], start=1):
                st.markdown(
                    (
                        f"{i}. **{item['label']}** | rrf={item.get('rrf_score', item['fusion_score']):.3f} "
                        f"(dense={item.get('dense_score', 0.0):.3f}, sparse={item.get('sparse_score', 0.0):.3f}, "
                        f"text={item['text_score']:.3f}, image={item['image_score']:.3f})"
                    )
                )
                st.caption(item["image_path"])

        st.session_state.messages.append({"role": "assistant", "content": out["answer"]})


if __name__ == "__main__":
    main()
