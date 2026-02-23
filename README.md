# AIML Project

An AI/ML pipeline for text summarization using clustering-based extractive methods.

## Project Structure

```
├── app.py                      # Main application entry point
├── requirements.txt            # Python dependencies
└── src/
    ├── preprocess.py           # Text cleaning and preprocessing
    ├── feature_extraction.py   # Feature/embedding extraction from text
    ├── clustering.py           # Sentence/document clustering
    ├── summarizer.py           # Summary generation logic
    └── utils.py                # Shared utility functions
```

## Getting Started

### Prerequisites

- Python 3.8++

### Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd AIML-Project
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

```bash
python app.py
```

## Pipeline Overview

1. **Preprocessing** – Clean and normalize raw text data.
2. **Feature Extraction** – Convert text into numerical representations (e.g., TF-IDF, embeddings).
3. **Clustering** – Group similar sentences/documents into clusters.
4. **Summarization** – Select representative sentences from each cluster to form a summary.

## License

This project is for educational and research purposes.
