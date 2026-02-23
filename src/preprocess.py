"""
Text preprocessing module.

Handles three main jobs:
  1. Downloading NLTK resources (punkt tokenizer, stopwords list).
  2. Cleaning raw text — collapsing whitespace, stripping junk chars.
  3. Splitting text into sentences and optionally tokenizing words.
"""

import re
import ssl
import functools

import nltk

# ── macOS SSL workaround ────────────────────────────────────────────
# Some macOS installs ship with outdated certs, which makes nltk.download
# fail over HTTPS.  This patches the default context so downloads work.
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass  # not on macOS — nothing to patch
else:
    ssl._create_default_https_context = _create_unverified_https_context

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

# ── NLTK resource management ───────────────────────────────────────

_NLTK_READY = False  # simple flag so we only check once per process


def ensure_nltk_data() -> None:
    """Download required NLTK data packages if they aren't already cached."""
    global _NLTK_READY
    if _NLTK_READY:
        return

    resources = {
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
        "stopwords": "corpora/stopwords",
    }
    for name, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)

    _NLTK_READY = True


# cache the stopword set so we don't rebuild it on every call
@functools.lru_cache(maxsize=1)
def _get_stop_words() -> frozenset:
    ensure_nltk_data()
    return frozenset(stopwords.words("english"))


# ── Text cleaning ──────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Collapse whitespace runs and strip leading/trailing spaces."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ── Sentence splitting ─────────────────────────────────────────────

def split_sentences(text: str) -> list[str]:
    """
    Split *text* into sentences using NLTK's Punkt tokenizer.

    Empty or whitespace-only fragments are dropped automatically.
    """
    ensure_nltk_data()
    sentences = sent_tokenize(text)
    # strip each sentence and throw away blanks
    return [s.strip() for s in sentences if s.strip()]


# ── Word-level tokenization ────────────────────────────────────────

def tokenize_and_clean(
    sentence: str,
    remove_stopwords: bool = True,
) -> list[str]:
    """
    Tokenize a sentence, lowercase it, keep only alphabetic tokens,
    and optionally drop English stopwords.

    Parameters
    ----------
    sentence : str
        A single sentence string.
    remove_stopwords : bool
        If True (default), common English stopwords are removed.

    Returns
    -------
    list[str]
        Cleaned token list, e.g. ["machine", "learning", "algorithms"].
    """
    ensure_nltk_data()
    tokens = word_tokenize(sentence.lower())
    tokens = [t for t in tokens if t.isalpha()]

    if remove_stopwords:
        stop_words = _get_stop_words()
        tokens = [t for t in tokens if t not in stop_words]

    return tokens


# ── Convenience pipeline ───────────────────────────────────────────

def preprocess_text(text: str) -> tuple[str, list[str]]:
    """
    Run the full preprocessing pipeline on raw input text.

    Steps:
      1. clean_text()   — normalise whitespace
      2. split_sentences() — NLTK Punkt sentence segmentation

    Returns
    -------
    tuple[str, list[str]]
        (cleaned_text, list_of_sentences)
    """
    cleaned = clean_text(text)
    sentences = split_sentences(cleaned)
    return cleaned, sentences