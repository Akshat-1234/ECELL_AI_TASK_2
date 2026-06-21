# E-CELL AI Task 2

This project implements a Retrieval-Augmented Generation (RAG) pipeline for answering questions from a collection of research papers. The system supports multiple retrieval and generation paths and includes basic evaluation metrics to compare them.

## Overview

The pipeline consists of:

- PDF parsing and text extraction
- Chunking with overlap
- Embedding generation using Sentence Transformers
- FAISS vector database for retrieval
- Three different answer generation pipelines:
  - API LLM (Gemini)
  - API LLM + Reranking
  - Local LLM (Ollama + Llama 3.2)
- FastAPI interface
- Evaluation framework for comparing different approaches

---

## Folder Structure

```
ECELL_AI_TASK_2
│
├── api/
│   └── app.py
│
├── data/
│   ├── pdfs/
│   └── vectorstore/
│
├── src/
│   ├── preprocess.py
│   ├── features.py
│   ├── train.py
│   ├── rerank.py
│   ├── localmodel.py
│   ├── metrics.py
│   ├── evaluate.py
│   └── utils.py
│
├── requirements.txt
├── REPORT.md
└── test_locals.py
```

---

## Pipeline

### 1. Document Processing

Research papers are loaded using PyMuPDF. Text is extracted page-wise and split into chunks using:

- Chunk size = 1000
- Overlap = 200

Page information is stored as metadata for later citation.

---

### 2. Embeddings and Vector Store

Embeddings are generated using:

```
sentence-transformers/all-MiniLM-L6-v2
```

The embeddings are stored locally using FAISS.

---

## Implemented Pipelines

### API LLM

```
Question
↓
FAISS Retrieval
↓
Gemini API
↓
Answer
```

Uses Gemini Flash to generate responses from retrieved chunks.

---

### API LLM + Rerank

```
Question
↓
FAISS Top 20
↓
Cross Encoder Reranker
↓
Top 5 Chunks
↓
Gemini API
↓
Answer
```

The reranker uses:

```
cross-encoder/ms-marco-MiniLM-L-6-v2
```

to improve retrieval quality before generation.

---

### Local LLM

```
Question
↓
FAISS Retrieval
↓
Llama 3.2 (Ollama)
↓
Answer
```

This path avoids external APIs and performs generation locally.

---

## FastAPI Interface

The project exposes a simple API.

Start the server:

```bash
uvicorn api.app:app --reload
```

Example request:

```json
{
    "query": "Explain self attention."
}
```

---

## Evaluation

The following metrics are implemented:

### Context Relevance (CR)

Measures how relevant retrieved chunks are to the query.

### Answer Relevance (AR)

Measures semantic similarity between the query and generated answer.

### Query Resolution Rate (QR)

Checks whether a query received a valid answer.

### Latency

Tracks retrieval and generation times.

---

## Running the Project

### Build chunks

```bash
python src/preprocess.py
```

### Create embeddings and vector database

```bash
python src/features.py
```

### Test the pipelines

```bash
python src/train.py
```

### Evaluate

```bash
python src/evaluate.py
```

### Start FastAPI

```bash
uvicorn api.app:app --reload
```

---

## Libraries Used

- LangChain
- FAISS
- Sentence Transformers
- HuggingFace Embeddings
- Gemini API
- Ollama
- FastAPI
- PyMuPDF
- Pandas
- Scikit-learn

---

## Notes

- Research papers are stored locally under `data/pdfs`.
- FAISS indices are serialized and stored under `data/vectorstore`.
- Local inference is performed using Ollama with Llama 3.2.
- Cross-encoder reranking is used to improve retrieval quality for complex queries.
- Evaluation results can be exported to CSV for comparison across pipelines.