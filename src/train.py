import os  # Directory Management
import time
from dotenv import load_dotenv

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_google_genai import ChatGoogleGenerativeAI

from src.utils import calculate_confidence, get_sources
from src.rerank import rerank_documents
from src.localmodel import generate_local_answer

# To safely load the APIKEY without leaking it
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Load embeddings (created in features.py)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS vector database
BASE_DIR = Path(__file__).parent.parent

vectorstore = FAISS.load_local(
    str(BASE_DIR / "data" / "vectorstore"),
    embeddings,
    allow_dangerous_deserialization=True
)

# We use Gemini FLASH as the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=api_key,
    temperature=0.3
)


def answer_question(question, k=10):

    # example question = "Compare BERT and GPT-3"

    # using similarity search we find the closest vectors to the question
    # vectorstore stores all the chunks
    # k determines how many nearest chunks are retrieved
    start = time.time()

    docs_and_scores = vectorstore.similarity_search_with_score(
        question,
        k=k
    )

    retrieval_time = float(time.time() - start)

    docs = []
    scores = []

    # separating the chunks and their similarity scores
    for doc, score in docs_and_scores:

        docs.append(doc)
        scores.append(score)

    # Calculating confidence score from similarity scores
    # Higher similarity means higher confidence
    if len(scores) == 0:

        return {
            "answer": "No relevant documents found.",
            "confidence": 0.0,
            "sources": [],
            "retrieval_time": retrieval_time,
            "generation_time": 0.0,
            "retrieved_docs": [],
        }

    confidence = calculate_confidence(scores)

    # If confidence is too low we avoid hallucinating
    if confidence < 0.3:

        return {
            "answer": "Insufficient information found in the documents.",
            "confidence": confidence,
            "sources": [],
            "retrieval_time": retrieval_time,
            "generation_time": 0.0,
            "retrieved_docs": docs,
        }

    # Adding source metadata
    # This will later help us display source file names and page numbers
    sources = get_sources(
        docs
    )

    # context
    # Combining all retrieved chunks into a single context string
    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    # prompt to llm to avoid hallucination
    prompt = f"""
Answer using ONLY the provided context.

If the answer is not contained in the context,
say that the information is unavailable.

Do not hallucinate.
Organize answers clearly.
Use bullet points when listing multiple items.
Provide a concise explanation for each item.

Mention source documents whenever relevant.

Context:
{context}

Question:
{question}
"""

    # sending prompt to Gemini and measuring generation time
    start = time.time()

    try:

        response = llm.invoke(prompt)

    # Gemini daily quota issues can interrupt execution
    except Exception:

        return {
            "answer": "Gemini API quota exceeded. Please try again later.",
            "confidence": confidence,
            "sources": sources,
            "retrieval_time": retrieval_time,
            "generation_time": 0.0,
            "retrieved_docs": docs,
        }

    generation_time = float(time.time() - start)

    return {
        "answer": response.content,
        "confidence": confidence,
        "sources": sources,
        "retrieval_time": retrieval_time,
        "generation_time": generation_time,
        "retrieved_docs": docs,
    }


# This function implements the reranked pipeline
# Instead of directly sending FAISS results to Gemini,
# we first retrieve more chunks and rerank them using a cross encoder

def answer_question_rerank(question):

    # First retrieve more chunks from FAISS
    # We intentionally retrieve 20 because reranking
    # will later select the best ones
    start = time.time()

    docs_and_scores = vectorstore.similarity_search_with_score(
        question,
        k=20
    )

    retrieval_time = float(
        time.time() - start
    )

    if len(docs_and_scores) == 0:

        return {
            "answer": "No relevant documents found.",
            "confidence": 0.0,
            "sources": [],
            "retrieval_time": retrieval_time,
            "generation_time": 0.0,
            "retrieved_docs": [],
        }

    # Using cross encoder reranking to improve relevance
    # docs here is the reranked top_k, NOT the raw FAISS top-20 —
    # this is what the LLM actually sees, and what should be used
    # for any downstream Context Relevance measurement.
    docs = rerank_documents(
        question,
        docs_and_scores,
        top_k=5
    )

    # Confidence is computed from the raw FAISS distances of the chunks
    # that survived reranking, so the rerank pipeline's output schema
    # stays consistent with the other two pipelines.
    raw_scores_by_content = {
        doc.page_content: score for doc, score in docs_and_scores
    }
    rerank_scores = [
        raw_scores_by_content[doc.page_content]
        for doc in docs
        if doc.page_content in raw_scores_by_content
    ]
    confidence = calculate_confidence(rerank_scores) if rerank_scores else 0.0

    # Source metadata for citation
    sources = get_sources(
        docs
    )

    # Combining chunks into context
    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    # Prompt to avoid hallucination
    prompt = f"""
Answer using ONLY the provided context.

If the answer is not contained in the context,
say that the information is unavailable.

Do not hallucinate.
Organize answers clearly.
Use bullet points when listing multiple items.

Context:
{context}

Question:
{question}
"""

    # Calling Gemini
    start = time.time()

    try:

        response = llm.invoke(
            prompt
        )

    except Exception:

        return {
            "answer": "Gemini API quota exceeded. Please try again later.",
            "confidence": confidence,
            "sources": sources,
            "retrieval_time": retrieval_time,
            "generation_time": 0.0,
            "retrieved_docs": docs,
        }

    generation_time = float(
        time.time() - start
    )

    return {
        "answer": response.content,
        "confidence": confidence,
        "sources": sources,
        "retrieval_time": retrieval_time,
        "generation_time": generation_time,
        "retrieved_docs": docs,
    }


# This function implements the Local LLM pipeline
# Retrieval is done by FAISS and answer generation is done locally using Ollama

def answer_question_local(question, k=5):

    start = time.time()

    # Retrieving relevant chunks
    docs_and_scores = vectorstore.similarity_search_with_score(
        question,
        k=k
    )

    retrieval_time = float(
        time.time() - start
    )

    docs = []
    scores = []

    for doc, score in docs_and_scores:

        docs.append(
            doc
        )
        scores.append(
            score
        )

    if len(scores) == 0:

        return {
            "answer": "No relevant documents found.",
            "confidence": 0.0,
            "sources": [],
            "retrieval_time": retrieval_time,
            "generation_time": 0.0,
            "retrieved_docs": [],
        }

    # Calculating confidence
    confidence = calculate_confidence(
        scores
    )

    # Same guardrail as answer_question() — if retrieval quality is too
    # low we skip the local LLM call entirely rather than risk a
    # hallucinated answer. Keeps the three pipelines consistent.
    if confidence < 0.3:

        return {
            "answer": "Insufficient information found in the documents.",
            "confidence": confidence,
            "sources": [],
            "retrieval_time": retrieval_time,
            "generation_time": 0.0,
            "retrieved_docs": docs,
        }

    # Source metadata for citations
    sources = get_sources(
        docs
    )

    # Combining retrieved chunks into context
    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    start = time.time()

    # Generating answer locally.
    # Ollama requires a separate running daemon — if it's not running,
    # or the model isn't pulled, this raises an exception rather than
    # crashing the whole evaluation loop.
    try:

        answer = generate_local_answer(
            context,
            question
        )

    except Exception:

        return {
            "answer": "Local LLM unavailable. Please ensure Ollama is running.",
            "confidence": confidence,
            "sources": sources,
            "retrieval_time": retrieval_time,
            "generation_time": 0.0,
            "retrieved_docs": docs,
        }

    generation_time = float(
        time.time() - start
    )

    return {
        "answer": answer,
        "confidence": confidence,
        "sources": sources,
        "retrieval_time": retrieval_time,
        "generation_time": generation_time,
        "retrieved_docs": docs,
    }


if __name__ == "__main__":

    result = answer_question_local(
        "Explain self attention."
    )

    print("\nAnswer:\n")

    print(
        result["answer"]
    )

    print("\nSources:\n")

    for source in result["sources"]:

        print(
            source["file"],
            "Page",
            source["page"]
        )