

import streamlit as st
from src.summarizer import summarize
from src.utils import SAMPLE_TEXTS, word_count, char_count

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Text Summarizer",
    page_icon="📝",
    layout="wide",
)

# ── Custom CSS ───────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem;
    }
    .main-header h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
    }
    .main-header p {
        color: #888;
        font-size: 1.1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-card h3 {
        margin: 0; font-size: 1.8rem; color: #4a4a8a;
    }
    .metric-card p {
        margin: 0; color: #666; font-size: 0.9rem;
    }
    .summary-box {
        background-color: #f0f4ff;
        border-left: 5px solid #667eea;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        font-size: 1.05rem;
        line-height: 1.7;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ───────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="main-header">
        <h1>📝 AI Text Summarizer</h1>
        <p>Extractive summarization using TF-IDF &amp; K-Means clustering</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

# ── Sidebar controls ────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    summary_ratio = st.slider(
        "Summary ratio",
        min_value=0.1,
        max_value=0.8,
        value=0.3,
        step=0.05,
        help="Fraction of original sentences to keep in the summary.",
    )

    st.divider()
    st.header("📚 Sample Texts")
    sample_choice = st.selectbox(
        "Load a sample",
        ["— Choose —"] + list(SAMPLE_TEXTS.keys()),
    )

    st.divider()
    st.markdown(
        """
        ### How it works
        1. **Preprocess** — clean & split into sentences
        2. **TF-IDF** — score sentence importance
        3. **K-Means** — cluster sentences by topic
        4. **Select** — pick the best from each cluster
        """
    )

# ── Main input area ─────────────────────────────────────────────────
if sample_choice and sample_choice != "— Choose —":
    default_text = SAMPLE_TEXTS[sample_choice]
else:
    default_text = ""

input_text = st.text_area(
    "Paste your text below:",
    value=default_text,
    height=250,
    placeholder="Enter or paste a paragraph / article here…",
)

# ── Summarize button ────────────────────────────────────────────────
col_btn, _ = st.columns([1, 4])
with col_btn:
    run = st.button("🚀 Summarize", type="primary", use_container_width=True)

if run:
    if not input_text or not input_text.strip():
        st.warning("Please enter some text to summarize.")
    else:
        with st.spinner("Analyzing and summarizing…"):
            result = summarize(input_text, ratio=summary_ratio)

        # Metrics row
        st.markdown("### 📊 Statistics")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(
                f'<div class="metric-card"><h3>{result["original_sentence_count"]}</h3>'
                f"<p>Original Sentences</p></div>",
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f'<div class="metric-card"><h3>{result["summary_sentence_count"]}</h3>'
                f"<p>Summary Sentences</p></div>",
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f'<div class="metric-card"><h3>{int(result["compression_ratio"] * 100)}%</h3>'
                f"<p>Compression Ratio</p></div>",
                unsafe_allow_html=True,
            )
        with m4:
            st.markdown(
                f'<div class="metric-card"><h3>{word_count(result["summary"])}</h3>'
                f"<p>Summary Words</p></div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # Summary output
        st.markdown("### ✨ Summary")
        st.markdown(
            f'<div class="summary-box">{result["summary"]}</div>',
            unsafe_allow_html=True,
        )

        # Side-by-side comparison
        st.markdown("---")
        st.markdown("### 🔍 Comparison")
        col_orig, col_summ = st.columns(2)
        with col_orig:
            st.markdown(f"**Original** — {word_count(input_text)} words, {char_count(input_text)} chars")
            st.info(input_text)
        with col_summ:
            st.markdown(
                f'**Summary** — {word_count(result["summary"])} words, '
                f'{char_count(result["summary"])} chars'
            )
            st.success(result["summary"])
