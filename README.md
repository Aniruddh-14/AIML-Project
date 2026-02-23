# AI Text Summarizer

An **extractive text summarizer** built with Python, powered by TF-IDF feature extraction and K-Means clustering. The app selects the most informative sentences from a document to create a concise summary — no neural networks needed.

> **Live demo:** Hosted on [Streamlit Community Cloud](https://aiml-project.streamlit.app)

---

## How It Works

The pipeline has four stages:

```
Raw Text  →  Preprocess  →  TF-IDF Features  →  K-Means Clustering  →  Summary
```

### 1. Preprocessing

- Collapse whitespace, strip noise characters.
- Split text into sentences using NLTK's Punkt tokenizer.
- Tokenize words, remove stopwords for downstream scoring.

### 2. Feature Extraction (TF-IDF)

Each sentence is converted into a numeric vector using **Term Frequency – Inverse Document Frequency**:

```
tf(t, d)    = count of term t in sentence d  /  total terms in d
idf(t, D)   = log( N / (1 + df(t)) ) + 1
tfidf(t, d) = tf(t, d)  ×  idf(t, D)
```

Where:

- `N` = total number of sentences in the document
- `df(t)` = number of sentences containing term `t`

We use **sublinear TF** (`1 + log(tf)`) to dampen the effect of very frequent words, and **L2 row-normalisation** so each sentence vector has unit length.

Each sentence gets an **importance score** = mean of its TF-IDF values. Higher score → more informative content.

### 3. K-Means Clustering

Sentences are grouped into `k` clusters, where `k = ratio × n_sentences`. K-Means minimises the **within-cluster sum of squares (WCSS)**:

```
J = Σ  Σ  ‖x - μ_k‖²
    k  x∈C_k
```

Where `μ_k` is the centroid of cluster `C_k`. This groups sentences that discuss similar topics together.

### 4. Representative Selection

From each cluster, pick the sentence with the **highest TF-IDF score**. Then sort the selected sentences by their original position to preserve narrative flow.

The result is a summary that covers all major topics proportionally while keeping the most informative phrasing.

---

## Project Structure

```
├── app.py                      # Streamlit web interface
├── requirements.txt            # Python dependencies
├── .streamlit/
│   └── config.toml             # Streamlit configuration
└── src/
    ├── __init__.py             # Package docstring
    ├── preprocess.py           # Text cleaning, sentence splitting
    ├── feature_extraction.py   # TF-IDF matrix + sentence scoring
    ├── clustering.py           # K-Means clustering + elbow method
    ├── summarizer.py           # Orchestrates the full pipeline
    └── utils.py                # Helpers and sample texts
```

---

## Getting Started

### Prerequisites

- Python 3.9+

### Installation

```bash
git clone https://github.com/the-carnage/AIML-Project.git
cd AIML-Project

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Run Locally

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Deploying on Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Click **New app** → select this repo → branch `main` → file `app.py`.
4. Click **Deploy**. That's it.

Streamlit Cloud auto-installs packages from `requirements.txt` and picks up `.streamlit/config.toml`.

---

## Tech Stack

- **Streamlit** — interactive web UI with dark/light theme
- **NLTK** — sentence tokenization and stopword lists
- **scikit-learn** — TF-IDF vectorization and K-Means clustering
- **NumPy** — numerical operations

---

## License

This project is for educational and research purposes.
