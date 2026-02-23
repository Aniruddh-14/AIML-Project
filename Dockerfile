# ── Base image ────────────────────────────────────────────
FROM python:3.10-slim

# ── System deps ───────────────────────────────────────────
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# ── Create a non-root user (Hugging Face Spaces requirement) ──
RUN useradd -m -u 1000 appuser

WORKDIR /app

# ── Install Python packages ───────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Pre-download NLTK data so the first request is fast ──
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"

# ── Copy application code ────────────────────────────────
COPY . .

# ── Permissions ───────────────────────────────────────────
RUN chown -R appuser:appuser /app
USER appuser

# ── Expose the Hugging Face Spaces port ──────────────────
EXPOSE 7860

# ── Health check (Spaces uses this to know the app is up) ─
HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health || exit 1

# ── Start Streamlit ──────────────────────────────────────
CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
