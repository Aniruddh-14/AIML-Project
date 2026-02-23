import streamlit as st
from src.summarizer import summarize
from src.utils import SAMPLE_TEXTS, word_count, char_count

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Text Summarizer",
    page_icon="📝",
    layout="wide",
)

# ── Theme state ──────────────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False


def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode


is_dark = st.session_state.dark_mode

# ── Theme colours ────────────────────────────────────────────────────
if is_dark:
    bg = "#0f1117"
    bg_secondary = "#1a1d2e"
    text_primary = "#e6e9f0"
    text_secondary = "#9ca3af"
    card_bg = "rgba(30, 34, 56, 0.7)"
    card_border = "rgba(102, 126, 234, 0.25)"
    card_value = "#a5b4fc"
    summary_bg = "rgba(30, 34, 56, 0.6)"
    summary_border = "#667eea"
    input_bg = "#1a1d2e"
    input_border = "#2d3148"
    input_text = "#e6e9f0"
    divider_color = "rgba(255,255,255,0.06)"
    sidebar_bg = "#141726"
    hover_glow = "rgba(102, 126, 234, 0.15)"
else:
    bg = "#f8f9fc"
    bg_secondary = "#ffffff"
    text_primary = "#1e1e2e"
    text_secondary = "#64748b"
    card_bg = "rgba(255, 255, 255, 0.7)"
    card_border = "rgba(102, 126, 234, 0.15)"
    card_value = "#4a4a8a"
    summary_bg = "rgba(240, 244, 255, 0.8)"
    summary_border = "#667eea"
    input_bg = "#ffffff"
    input_border = "#e2e8f0"
    input_text = "#1e1e2e"
    divider_color = "rgba(0,0,0,0.06)"
    sidebar_bg = "#f1f3f8"
    hover_glow = "rgba(102, 126, 234, 0.08)"

# ── Custom CSS ───────────────────────────────────────────────────────
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* ─── Global ─── */
    html, body, .stApp {{
        font-family: 'Inter', sans-serif;
        background-color: {bg};
        color: {text_primary};
        transition: background-color 0.4s ease, color 0.4s ease;
    }}

    .stApp > header {{
        background: transparent !important;
    }}

    /* ─── Sidebar ─── */
    section[data-testid="stSidebar"] {{
        background-color: {sidebar_bg} !important;
        transition: background-color 0.4s ease;
    }}
    section[data-testid="stSidebar"] * {{
        color: {text_primary} !important;
        transition: color 0.4s ease;
    }}

    /* ─── Animated header ─── */
    .main-header {{
        text-align: center;
        padding: 2rem 0 1rem;
    }}
    .main-header h1 {{
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #667eea, #764ba2, #f093fb, #667eea);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        animation: gradientShift 6s ease infinite;
    }}
    .main-header p {{
        color: {text_secondary};
        font-size: 1.1rem;
        font-weight: 400;
        margin-top: 0.3rem;
        letter-spacing: 0.01em;
    }}
    @keyframes gradientShift {{
        0%   {{ background-position: 0% 50%; }}
        50%  {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}

    /* ─── Metric cards (glassmorphism) ─── */
    .metric-card {{
        background: {card_bg};
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid {card_border};
        border-radius: 16px;
        padding: 1.4rem 1rem;
        text-align: center;
        transition: transform 0.25s ease, box-shadow 0.25s ease, background 0.4s ease;
    }}
    .metric-card:hover {{
        transform: translateY(-4px);
        box-shadow: 0 8px 30px {hover_glow};
    }}
    .metric-card h3 {{
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        color: {card_value};
        transition: color 0.4s ease;
    }}
    .metric-card p {{
        margin: 0.3rem 0 0;
        color: {text_secondary};
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        transition: color 0.4s ease;
    }}

    /* ─── Summary box ─── */
    .summary-box {{
        background: {summary_bg};
        backdrop-filter: blur(10px);
        border-left: 4px solid {summary_border};
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        font-size: 1.05rem;
        line-height: 1.8;
        color: {text_primary};
        transition: background 0.4s ease, color 0.4s ease;
    }}

    /* ─── Theme toggle button ─── */
    .theme-toggle {{
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.6rem;
        padding: 0.6rem 1.2rem;
        border-radius: 50px;
        background: linear-gradient(135deg, {card_bg}, {card_bg});
        border: 1px solid {card_border};
        cursor: pointer;
        font-size: 0.95rem;
        font-weight: 600;
        color: {text_primary};
        transition: all 0.3s ease;
        margin: 0 auto 0.5rem;
        width: fit-content;
    }}
    .theme-toggle:hover {{
        box-shadow: 0 4px 20px {hover_glow};
        transform: scale(1.03);
    }}

    /* ─── Divider ─── */
    hr {{
        border-color: {divider_color} !important;
    }}

    /* ─── Text area ─── */
    .stTextArea textarea {{
        background-color: {input_bg} !important;
        color: {input_text} !important;
        border-color: {input_border} !important;
        border-radius: 12px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
        transition: background-color 0.4s ease, color 0.4s ease, border-color 0.4s ease;
    }}
    .stTextArea textarea:focus {{
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.15) !important;
    }}

    /* ─── Buttons ─── */
    .stButton > button[kind="primary"] {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        letter-spacing: 0.02em !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    }}
    .stButton > button[kind="primary"]:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 25px rgba(102, 126, 234, 0.35) !important;
    }}

    /* ─── Section headings ─── */
    .section-title {{
        font-family: 'Inter', sans-serif;
        font-size: 1.3rem;
        font-weight: 700;
        color: {text_primary};
        margin: 1.5rem 0 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}

    /* ─── How-it-works steps ─── */
    .step-item {{
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
        padding: 0.7rem 0;
    }}
    .step-num {{
        flex-shrink: 0;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.8rem;
        font-weight: 700;
    }}
    .step-text {{
        font-size: 0.9rem;
        color: {text_secondary};
        line-height: 1.5;
    }}
    .step-text strong {{
        color: {text_primary};
    }}

    /* ─── Footer ─── */
    .app-footer {{
        text-align: center;
        padding: 2rem 0 1rem;
        color: {text_secondary};
        font-size: 0.82rem;
        letter-spacing: 0.02em;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ───────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="main-header">
        <h1>📝 AI Text Summarizer</h1>
        <p>Extractive summarization powered by TF-IDF &amp; K-Means clustering</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

# ── Sidebar controls ────────────────────────────────────────────────
with st.sidebar:
    # ── Theme toggle ──
    theme_icon = "☀️" if is_dark else "🌙"
    theme_label = "Light Mode" if is_dark else "Dark Mode"
    st.button(
        f"{theme_icon}  {theme_label}",
        on_click=toggle_theme,
        use_container_width=True,
    )
    st.divider()

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
        f"""
        <div style="margin-top:0.5rem;">
            <div class="step-item">
                <div class="step-num">1</div>
                <div class="step-text"><strong>Preprocess</strong> — clean &amp; split into sentences</div>
            </div>
            <div class="step-item">
                <div class="step-num">2</div>
                <div class="step-text"><strong>TF-IDF</strong> — score sentence importance</div>
            </div>
            <div class="step-item">
                <div class="step-num">3</div>
                <div class="step-text"><strong>K-Means</strong> — cluster sentences by topic</div>
            </div>
            <div class="step-item">
                <div class="step-num">4</div>
                <div class="step-text"><strong>Select</strong> — pick the best from each cluster</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
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
        st.markdown('<div class="section-title">📊 Statistics</div>', unsafe_allow_html=True)
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
                f"<p>Compression</p></div>",
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
        st.markdown('<div class="section-title">✨ Summary</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="summary-box">{result["summary"]}</div>',
            unsafe_allow_html=True,
        )

        # Side-by-side comparison
        st.markdown("---")
        st.markdown('<div class="section-title">🔍 Comparison</div>', unsafe_allow_html=True)
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

    # Footer
    st.markdown(
        '<div class="app-footer">Built with Streamlit · TF-IDF · K-Means Clustering</div>',
        unsafe_allow_html=True,
    )
