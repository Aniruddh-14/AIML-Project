"""
Main summarizer module.

Orchestrates the full extractive summarization pipeline:

    raw text  →  preprocess  →  TF-IDF features  →  K-Means clustering  →  summary

This is the only module the rest of the app needs to import.
"""

from __future__ import annotations

from src.preprocess import preprocess_text
from src.feature_extraction import build_tfidf_matrix, get_sentence_scores
from src.clustering import cluster_sentences, select_representative_sentences


def summarize(text: str, ratio: float = 0.3) -> dict:
    """
    Produce an extractive summary of *text*.

    The algorithm:
      1. Clean text and split into sentences.
      2. Build a TF-IDF matrix (each row = one sentence vector).
      3. Score sentences by mean TF-IDF value.
      4. Cluster sentences into *k* groups using K-Means.
      5. From each cluster, pick the highest-scoring sentence.
      6. Return those representatives in document order.

    Parameters
    ----------
    text  : str
        The raw input text to summarise.
    ratio : float, default 0.3
        Fraction of original sentences to keep (0.0 – 1.0).
        For example, 0.3 means "keep roughly 30 % of sentences".

    Returns
    -------
    dict with keys
        summary                : str   — the generated summary
        original_sentence_count: int   — sentences in the input
        summary_sentence_count : int   — sentences in the summary
        compression_ratio      : float — summary / original (lower = more compressed)
    """
    # ── guard: empty or near-empty input ──────────────────────────
    if not text or not text.strip():
        return _empty_result()

    cleaned, sentences = preprocess_text(text)

    # if there are only a couple of sentences, just return the whole thing
    if len(sentences) <= 2:
        return {
            "summary": cleaned,
            "original_sentence_count": len(sentences),
            "summary_sentence_count": len(sentences),
            "compression_ratio": 1.0,
        }

    # ── determine how many clusters / summary sentences we want ──
    n_clusters = max(1, int(len(sentences) * ratio))
    n_clusters = min(n_clusters, len(sentences))

    # ── feature extraction ────────────────────────────────────────
    tfidf_matrix, _vectorizer = build_tfidf_matrix(sentences)
    scores = get_sentence_scores(tfidf_matrix)

    # ── clustering + representative selection ─────────────────────
    labels = cluster_sentences(tfidf_matrix, n_clusters)
    summary_sentences = select_representative_sentences(sentences, labels, scores)

    summary_text = " ".join(summary_sentences)

    return {
        "summary": summary_text,
        "original_sentence_count": len(sentences),
        "summary_sentence_count": len(summary_sentences),
        "compression_ratio": round(len(summary_sentences) / len(sentences), 2),
    }


# ── helpers ────────────────────────────────────────────────────────

def _empty_result() -> dict:
    """Return a safe default when there's nothing to summarise."""
    return {
        "summary": "",
        "original_sentence_count": 0,
        "summary_sentence_count": 0,
        "compression_ratio": 0.0,
    }