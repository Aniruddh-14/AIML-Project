"""
Feature extraction module.

Converts a list of sentences into a TF-IDF matrix and derives per-sentence
importance scores.

Math refresher
--------------
TF-IDF stands for *Term Frequency – Inverse Document Frequency*.

For a term *t* in sentence (document) *d* within a corpus *D* of *N* docs:

    tf(t, d)  = count(t in d) / |d|           — how often the word appears
    idf(t, D) = log( N / (1 + df(t, D)) ) + 1 — rarity across all docs

    tfidf(t, d) = tf(t, d) * idf(t, D)

Scikit-learn's TfidfVectorizer applies L2 row-normalisation by default so
each sentence vector has unit length.  This lets us use the dot product
directly as cosine similarity later on.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as _cosine_similarity


def build_tfidf_matrix(
    sentences: list[str],
) -> tuple[np.ndarray, TfidfVectorizer]:
    """
    Build a TF-IDF matrix from a list of sentence strings.

    Parameters
    ----------
    sentences : list[str]
        Raw sentence strings (stop-word removal is handled internally).

    Returns
    -------
    tfidf_matrix : sparse CSR matrix, shape (n_sentences, n_features)
    vectorizer   : fitted TfidfVectorizer (useful for inspection / vocab)
    """
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_df=0.95,        # ignore terms appearing in >95 % of sentences
        min_df=1,           # keep singletons — sentences are short
        sublinear_tf=True,  # apply 1 + log(tf) dampening
    )
    tfidf_matrix = vectorizer.fit_transform(sentences)
    return tfidf_matrix, vectorizer


def get_sentence_scores(tfidf_matrix) -> np.ndarray:
    """
    Score each sentence by the mean TF-IDF value across its terms.

    A higher score indicates that the sentence contains more *informative*
    (i.e., less common) words relative to the rest of the document.

    Parameters
    ----------
    tfidf_matrix : sparse matrix, shape (n_sentences, n_features)

    Returns
    -------
    scores : ndarray, shape (n_sentences,)
    """
    # .mean(axis=1) returns a matrix; flatten to 1-D
    scores = np.asarray(tfidf_matrix.mean(axis=1)).flatten()
    return scores


def compute_similarity_matrix(tfidf_matrix) -> np.ndarray:
    """
    Compute pairwise cosine similarity between all sentence vectors.

    Because the TF-IDF rows are L2-normalised, cosine similarity
    simplifies to the dot product:

        sim(s_i, s_j) = s_i · s_j

    Returns
    -------
    sim_matrix : ndarray, shape (n_sentences, n_sentences)
        Values in [0, 1] — 1 means identical direction.
    """
    return _cosine_similarity(tfidf_matrix)