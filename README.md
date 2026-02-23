---
title: AI Text Summarizer
emoji: 📝
colorFrom: indigo
colorTo: blue
sdk: docker
app_file: app.py
pinned: false
license: mit
---

# AI Text Summarizer

An **extractive text summarizer** built with Python, powered by TF-IDF feature extraction and K-Means clustering. The app selects the most informative sentences from a document to create a concise summary — no neural networks needed.

> **Live demo:** Hosted on [Hugging Face Spaces](https://huggingface.co/spaces/the-carnage/AIML-Project)

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
├── app.py                          # Streamlit web interface
├── Dockerfile                      # Docker config for HF Spaces
├── requirements.txt                # Pinned Python dependencies
├── .github/
│   └── workflows/
│       ├── ci.yml                  # Lint + smoke tests on every push
│       └── sync_to_hf_space.yml   # Auto-deploy to Hugging Face Spaces
└── src/
    ├── __init__.py                 # Package init with top-level import
    ├── preprocess.py               # Text cleaning, sentence splitting
    ├── feature_extraction.py       # TF-IDF matrix + sentence scoring
    ├── clustering.py               # K-Means clustering + elbow method
    ├── summarizer.py               # Orchestrates the full pipeline
    └── utils.py                    # Helpers and sample texts
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

## CI/CD

| Workflow                             | Trigger             | What it does                             |
| ------------------------------------ | ------------------- | ---------------------------------------- |
| **Lint & Test** (`ci.yml`)           | Push / PR to `main` | Runs flake8 + smoke tests                |
| **HF Sync** (`sync_to_hf_space.yml`) | Push to `main`      | Force-pushes code to Hugging Face Spaces |

### Required GitHub Secrets (for HF deploy)

| Secret        | Description                                  |
| ------------- | -------------------------------------------- |
| `HF_TOKEN`    | Hugging Face token with **Write** permission |
| `HF_USERNAME` | Your HF username (e.g. `the-carnage`)        |
| `HF_SPACE`    | Space repo name (e.g. `AIML-Project`)        |

---

## Tech Stack

- **Streamlit** — interactive web UI with dark/light theme
- **NLTK** — sentence tokenization and stopword lists
- **scikit-learn** — TF-IDF vectorization and K-Means clustering
- **NumPy** — numerical operations
- **Docker** — containerised deployment on Hugging Face Spaces

---

## License

This project is for educational and research purposes.
