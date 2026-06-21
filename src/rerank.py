# Cross encoder is used to rerank the chunks retrieved by FAISS
# Unlike embeddings, it directly looks at (query,chunk) pairs
# and produces a relevance score

from sentence_transformers import CrossEncoder

# This model is commonly used for document reranking
reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

def rerank_documents(question, docs_and_scores, top_k=5):

    # docs_and_scores contains tuples of (document,distance)
    # We only need the document contents for reranking

    pairs = []

    for doc, _ in docs_and_scores:

        pairs.append(
            (
                question,
                doc.page_content
            )
        )

    # Cross encoder predicts a relevance score
    rerank_scores = reranker.predict(
        pairs
    )

    ranked_docs = []

    # Combining documents with their rerank scores
    for i, (doc, _) in enumerate(docs_and_scores):

        ranked_docs.append(
            (
                doc,
                rerank_scores[i]
            )
        )

    # Sorting in descending order of score
    ranked_docs.sort(
        key=lambda x: x[1],
        reverse=True
    )

    # Selecting only top_k documents after reranking
    final_docs = []

    for doc, _ in ranked_docs[:top_k]:

        final_docs.append(
            doc
        )

    return final_docs