"""
Streamlit Application: Topic Modeling with Latent Dirichlet Allocation (LDA)
Discovering Topics and Their Top Words in a Small News Document Collection.
Matches Lab Exercise 4 Specification.
"""

import os
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.preprocess import Preprocessor
from src.topic_model import LDATopicModel
from src.metrics import compute_npmi_coherence, compute_clustering_agreement

st.set_page_config(
    page_title="LDA Topic Modeling",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Contrast Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .subtitle {
        color: #94A3B8;
        font-size: 1.05rem;
        margin-bottom: 24px;
    }
    .kpi-container {
        display: flex;
        gap: 20px;
        margin-bottom: 20px;
    }
    .kpi-box {
        background-color: #1E293B;
        border-radius: 8px;
        padding: 16px 20px;
        flex: 1;
        text-align: left;
    }
    .kpi-label {
        color: #94A3B8;
        font-size: 0.85rem;
        margin-bottom: 4px;
    }
    .kpi-value {
        color: #F8FAFC;
        font-size: 2rem;
        font-weight: 700;
    }
    .success-banner {
        background-color: #064E3B;
        color: #A7F3D0;
        padding: 12px 18px;
        border-radius: 6px;
        border: 1px solid #059669;
        font-size: 0.95rem;
        margin-bottom: 25px;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        margin-top: 30px;
        margin-bottom: 12px;
        color: #F8FAFC;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------
with st.sidebar:
    st.subheader("Corpus")
    corpus_source = st.radio(
        "Documents",
        ["Bundled BBC sample (100 articles)", "Paste text", "Upload files"],
        index=0
    )

    st.subheader("Model")
    n_topics = st.slider("Number of topics (K)", min_value=2, max_value=10, value=5)
    n_top_words = st.slider("Top words per topic", min_value=3, max_value=15, value=5)
    n_restarts = st.slider("Random restarts (keep best likelihood)", min_value=1, max_value=20, value=10)

    with st.expander("Preprocessing & vocabulary"):
        min_df_val = st.number_input("min_df (drop rare words)", min_value=1, max_value=10, value=5)
        max_df_val = st.slider("max_df (drop corpus-wide words)", min_value=0.1, max_value=0.9, value=0.3, step=0.05)
        use_lemmatize = st.checkbox("WordNet Lemmatisation", value=True)
        use_news_stops = st.checkbox("Remove News Stop Words", value=True)


# ---------------------------------------------------------
# Data Loading
# ---------------------------------------------------------
@st.cache_data
def load_bundled_data():
    csv_file = "data/bbc_sample_100.csv"
    if os.path.exists(csv_file):
        return pd.read_csv(csv_file)
    return pd.DataFrame({"doc_id": range(1, 6), "category": ["news"]*5, "text": ["sample text"]*5})

if corpus_source == "Bundled BBC sample (100 articles)":
    df_corpus = load_bundled_data()
    docs = df_corpus["text"].tolist()
    labels = df_corpus["category"].tolist() if "category" in df_corpus.columns else None
elif corpus_source == "Paste text":
    user_pasted = st.text_area("Paste articles (separated by blank lines):", height=150)
    docs = [d.strip() for d in user_pasted.split("\n\n") if len(d.strip()) > 10]
    if not docs:
        docs = ["Please paste text documents separated by double blank lines."]
    labels = None
    df_corpus = pd.DataFrame({"text": docs})
else:
    uploaded = st.file_uploader("Upload CSV or TXT file:", type=["csv", "txt"])
    if uploaded is not None:
        if uploaded.name.endswith(".csv"):
            df_up = pd.read_csv(uploaded)
            col = [c for c in df_up.columns if "text" in c.lower()][0] if any("text" in c.lower() for c in df_up.columns) else df_up.columns[0]
            docs = df_up[col].dropna().astype(str).tolist()
            labels = df_up["category"].tolist() if "category" in df_up.columns else None
            df_corpus = df_up
        else:
            content = uploaded.read().decode("utf-8", errors="ignore")
            docs = [d.strip() for d in content.split("\n\n") if len(d.strip()) > 10]
            labels = None
            df_corpus = pd.DataFrame({"text": docs})
    else:
        df_corpus = load_bundled_data()
        docs = df_corpus["text"].tolist()
        labels = df_corpus["category"].tolist()


# ---------------------------------------------------------
# Top Header & Summary KPIs
# ---------------------------------------------------------
st.markdown('<div class="main-title">✨ LDA Topic Modeling</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Discover the hidden topics in a collection of documents with Latent Dirichlet Allocation, and see the top words that describe each topic.</div>', unsafe_allow_html=True)

raw_words_count = sum(len(d.split()) for d in docs)
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f'<div class="kpi-box"><div class="kpi-label">Documents</div><div class="kpi-value">{len(docs)}</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="kpi-box"><div class="kpi-label">Words (raw)</div><div class="kpi-value">{raw_words_count:,}</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="kpi-box"><div class="kpi-label">Topics requested</div><div class="kpi-value">{n_topics}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
discover_clicked = st.button("🔴 Discover Topics", type="primary", use_container_width=False)


# ---------------------------------------------------------
# LDA Execution & Model Caching
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def train_lda_model(docs_tuple, k, min_df, max_df, restarts, lemmatize):
    prep = Preprocessor(lemmatize=lemmatize)
    model = LDATopicModel(
        n_components=k,
        min_df=min_df,
        max_df=max_df,
        n_restarts=restarts,
        random_state=0,
        max_iter=100,
        preprocessor=prep
    )
    model.fit(list(docs_tuple))
    return model, prep

if discover_clicked or "model_fitted" in st.session_state:
    st.session_state["model_fitted"] = True

    with st.spinner("Fitting LDA topic model..."):
        model, prep = train_lda_model(tuple(docs), n_topics, min_df_val, max_df_val, n_restarts, use_lemmatize)

    vocab_len = len(model.vocab)
    token_sum = model.X.sum()
    seed_chosen = model.best_seed
    perplexity_val = round(model.perplexity)

    st.markdown(f"""
    <div class="success-banner">
        ✔ Fitted LDA with K = {n_topics} on {len(docs)} documents: vocabulary {vocab_len} words, {token_sum:,} tokens; best of {n_restarts} restart(s) (seed {seed_chosen}) — perplexity {perplexity_val}
    </div>
    """, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # 1. Top 5 words per topic
    # ---------------------------------------------------------
    st.markdown(f'<div class="section-header">1. Top {n_top_words} words per topic</div>', unsafe_allow_html=True)
    top_words = model.get_top_words(n=n_top_words)
    dominant_topics = model.get_dominant_topics()
    doc_counts = pd.Series(dominant_topics).value_counts().to_dict()

    topic_rows = []
    for k in range(n_topics):
        words_only = ", ".join([w for w, _ in top_words[k]])
        weights_only = ", ".join([f"{p:.3f}" for _, p in top_words[k]])
        topic_rows.append({
            "Topic": f"Topic {k + 1}",
            "Top words": words_only,
            "Weights": weights_only,
            "Docs (dominant)": doc_counts.get(k, 0)
        })

    df_top_words = pd.DataFrame(topic_rows)
    st.dataframe(df_top_words, use_container_width=True, hide_index=True)

    csv_topics = df_top_words.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download topics (CSV)",
        data=csv_topics,
        file_name="lda_topics.csv",
        mime="text/csv"
    )

    # ---------------------------------------------------------
    # 2. Word weights by topic (Bar charts)
    # ---------------------------------------------------------
    st.markdown('<div class="section-header">2. Word weights by topic</div>', unsafe_allow_html=True)
    bar_cols = st.columns(n_topics)
    colors = ["#4F46E5", "#F59E0B", "#10B981", "#EF4444", "#06B6D4", "#8B5CF6", "#EC4899", "#84CC16", "#F97316", "#14B8A6"]

    for k in range(n_topics):
        with bar_cols[k]:
            items = top_words[k]
            w_names = [w for w, _ in reversed(items)]
            w_probs = [p for _, p in reversed(items)]
            
            fig = go.Figure(go.Bar(
                x=w_probs,
                y=w_names,
                orientation='h',
                marker_color=colors[k % len(colors)],
                text=[f"{p:.3f}" for p in w_probs],
                textposition='inside',
                insidetextanchor='middle'
            ))
            fig.update_layout(
                title=f"<b>Topic {k + 1}</b>",
                margin=dict(l=10, r=10, t=35, b=10),
                height=260,
                xaxis=dict(showgrid=False, showticklabels=False),
                yaxis=dict(showgrid=False),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------
    # 3. Topic mixture of each document
    # ---------------------------------------------------------
    st.markdown('<div class="section-header">3. Topic mixture of each document</div>', unsafe_allow_html=True)
    doc_mixture_rows = []
    for i, doc in enumerate(docs):
        dom_k = dominant_topics[i] + 1
        dom_p = model.theta[i, dominant_topics[i]]
        row = {
            "Document": doc[:90] + "...",
            "Label": labels[i] if labels else "N/A",
            "Dominant topic": dom_k,
            "Probability": round(float(dom_p), 3)
        }
        for k in range(n_topics):
            row[f"T{k + 1}"] = round(float(model.theta[i, k]), 3)
        doc_mixture_rows.append(row)

    df_doc_mix = pd.DataFrame(doc_mixture_rows)
    st.dataframe(df_doc_mix, use_container_width=True, hide_index=True, height=280)

    csv_doc_mix = df_doc_mix.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download document topics (CSV)",
        data=csv_doc_mix,
        file_name="document_topics.csv",
        mime="text/csv"
    )

    # ---------------------------------------------------------
    # 4. Topic prevalence vs known categories
    # ---------------------------------------------------------
    st.markdown('<div class="section-header">4. Topic prevalence vs. known categories</div>', unsafe_allow_html=True)
    col_p1, col_p2 = st.columns(2)

    with col_p1:
        prev_vals = model.get_topic_prevalence()
        df_prev = pd.DataFrame({
            "Topic": [f"Topic {k + 1}" for k in range(n_topics)],
            "Average share": prev_vals
        })
        fig_prev = px.bar(
            df_prev,
            x="Topic",
            y="Average share",
            color="Topic",
            color_discrete_sequence=colors,
            title="<b>Topic Prevalence (Average Document Share)</b>"
        )
        fig_prev.update_layout(template="plotly_dark", height=380, showlegend=False)
        st.plotly_chart(fig_prev, use_container_width=True)

    with col_p2:
        if labels:
            df_cat = pd.DataFrame({
                "Category": labels,
                "Dominant topic": [f"Topic {t + 1}" for t in dominant_topics]
            })
            ct = pd.crosstab(df_cat["Category"], df_cat["Dominant topic"])
            fig_heat = px.imshow(
                ct,
                text_auto=True,
                color_continuous_scale="Blues",
                title="<b>Category × Dominant Topic Heat Map</b>",
                labels=dict(x="Dominant topic", y="Category", color="Count")
            )
            fig_heat.update_layout(template="plotly_dark", height=380)
            st.plotly_chart(fig_heat, use_container_width=True)
            st.caption("Labels are used only for this check — LDA never sees them.")
        else:
            st.info("Upload or bundle categorized data to view the Category × Dominant Topic heat map.")
