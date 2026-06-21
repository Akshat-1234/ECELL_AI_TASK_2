# This file contains evaluation metrics used to judge retrieval quality

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# We use the same embedding model for metric computation
embedder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# Context Relevance (CR)
# Measures how relevant the retrieved chunks are to the query
def context_relevance(question, docs):

    # Query embedding
    query_embedding = embedder.encode(
        [question]
    )

    # Chunk embeddings
    doc_embeddings = embedder.encode(
        [
            doc.page_content
            for doc in docs
        ]
    )

    # Average cosine similarity between query and chunks
    similarities = cosine_similarity(
        query_embedding,
        doc_embeddings
    )

    return float(
        np.mean(similarities)
    )


# Answer Relevance (AR)
# Measures whether the generated answer is related to the question
def answer_relevance(question, answer):

    query_embedding = embedder.encode(
        [question]
    )

    answer_embedding = embedder.encode(
        [answer]
    )

    similarity = cosine_similarity(
        query_embedding,
        answer_embedding
    )

    return float(
        similarity[0][0]
    )


# Query Resolution Rate (QR)
# Success = answer is not empty and not insufficient
def query_resolved(answer):

    if answer is None:

        return 0

    answer = answer.lower()

    if "insufficient information" in answer:

        return 0

    if "quota exceeded" in answer:

        return 0

    return 1