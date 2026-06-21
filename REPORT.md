# Research Paper RAG Assistant

This project implements a Retrieval-Augmented Generation (RAG) pipeline for answering questions from a collection of machine learning research papers. Instead of relying solely on the knowledge stored inside an LLM, the system first retrieves relevant information from papers and then generates answers based only on the retrieved context.

The goal of this project was to build a simple but complete RAG system that supports source citation, confidence estimation, evaluation, and an API interface.

---

## Dataset

The corpus consists of several well-known machine learning and NLP papers:

- Attention Is All You Need
- BERT
- GPT-3
- Llama 2
- Sentence-BERT
- LoRA: Low-Rank Adaptation of Large Language Models
- Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

These papers were chosen because they cover different aspects of modern NLP and allow both single-document and cross-document reasoning.

---

## Project Structure

```
ECELL_AI_TASK_2
│
├── api
│   └── app.py
│
├── .env
│
├── data
│   ├── pdfs
│   └── vectorstore
│
├── src
│   ├── preprocess.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   ├── utils.py
│   └── __init__.py
│
├── requirements.txt
└── README.md
```

---

## Pipeline

The overall workflow is fairly straightforward:

```
PDFs
↓
Text Extraction
↓
Chunking
↓
Sentence Embeddings
↓
FAISS Vector Database
↓
Similarity Search
↓
Gemini
↓
Answer + Sources + Confidence
```

---

## Document Processing

PDFs are processed using **PyMuPDF**.

Instead of treating each paper as one large document, text is extracted page by page. This allows page numbers to be preserved, making source attribution easier later.

The text is split using `RecursiveCharacterTextSplitter`.

Parameters used:

- Chunk size = 1000
- Chunk overlap = 200

This overlap helps preserve context between neighboring chunks.

A total of 1056 chunks were generated from the paper collection.

---

## Embeddings and Vector Store

Each chunk is converted into embeddings using:

```
sentence-transformers/all-MiniLM-L6-v2
```

The embeddings are stored inside a FAISS vector database, which enables efficient semantic search over the document collection.

---

## Retrieval and Generation

When a query is received:

1. The query is converted into an embedding.
2. FAISS retrieves the most relevant chunks.
3. Retrieved chunks are combined into a context.
4. Gemini 2.5 Flash Lite generates an answer.
5. Source papers and page numbers are returned along with the response.

To reduce hallucinations, the model is instructed to answer only using the provided context.

---

## Evaluation

Different values of **k** were tested:

- k = 3
- k = 5
- k = 10
- k = 20

Increasing k improved recall but also increased latency and occasionally introduced irrelevant context.

Among the tested values, **k = 5** gave the best balance between retrieval quality and response time.

---

## Example Queries

### Single Paper Queries

- Explain self attention.
- What is masked language modeling?
- What is Low Rank Adaptation?
- What is in-context learning?

### Cross Paper Queries

- Compare BERT and GPT-3.
- Compare GPT-3 and Llama 2.
- How does RAG address limitations of GPT-3?
- Why is LoRA useful for fine-tuning Llama models?
- How have large language models evolved from Transformers to GPT-3 and Llama 2?

These queries test the model's ability to synthesize information across multiple papers rather than simply retrieve isolated facts.

---

## API

The project uses FastAPI to expose the pipeline through a REST API.

### Endpoint

```
POST /query
```

Request:

```json
{
  "query": "Explain self attention."
}
```

Response:

```json
{
  "answer": "...",
  "confidence": 0.89,
  "sources": [
    {
      "file": "Attention Is All You Need.pdf",
      "page": 3
    }
  ],
  "retrieval_time": 0.02,
  "generation_time": 1.84
}
```

Swagger documentation can be accessed at:

```
http://127.0.0.1:8000/docs
```

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd ECELL_AI_TASK_2
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Generate chunks

```bash
python src/preprocess.py
```

### Build vector database

```bash
python src/features.py
```

### Test retrieval pipeline

```bash
python src/train.py
```

### Run evaluation

```bash
python src/evaluate.py
```

### Start API

```bash
uvicorn api.app:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

---

## Limitations

The confidence score is derived from similarity scores and should not be interpreted as answer correctness.

The system currently uses only dense retrieval and does not employ reranking methods.

Generation quality is also dependent on Gemini API availability and quota limits.

---

## Future Improvements

Some possible extensions include:

- Hybrid search using BM25 and dense embeddings.
- Cross-encoder reranking.
- RAGAS evaluation metrics.
- Query rewriting and expansion.
- Agentic RAG workflows.
- Support for larger document collections.

---

## Final Thoughts

This project was built as an attempt to understand the complete RAG pipeline rather than simply use a high-level framework. While the system itself is fairly simple, building it from scratch helped in understanding document preprocessing, embeddings, vector databases, retrieval strategies, prompt design, and API deployment. It also highlighted some practical challenges such as PDF extraction issues, retrieval quality trade-offs, and API limitations.