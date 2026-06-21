import os #Directory Management
import time
from dotenv import load_dotenv

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_google_genai import ChatGoogleGenerativeAI

from src.utils import calculate_confidence,get_sources

#To safely load the APIKEY without leaking it
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

#Load embeddings(created in features.py)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

#Load FAISS vector database
BASE_DIR = Path(__file__).parent.parent

vectorstore = FAISS.load_local(
    str(BASE_DIR / "data" / "vectorstore"),
    embeddings,
    allow_dangerous_deserialization=True
)

#We use Gemini FLASH as the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=api_key,
    temperature=0.3
)

def answer_question(question,k=10):

    #example question = "Compare BERT and GPT-3"

    #using similarity search we find the closest vectors to the question
    #vectorstore stores all the chunks
    #k determines how many nearest chunks are retrieved
    start = time.time()

    docs_and_scores = vectorstore.similarity_search_with_score(
        question,
        k=k
    )

    retrieval_time = float(time.time() - start)

    docs = []
    scores = []

    #separating the chunks and their similarity scores
    for doc,score in docs_and_scores:

        docs.append(doc)
        scores.append(score)

    #Calculating confidence score from similarity scores
    #Higher similarity means higher confidence
    if len(scores) == 0:

        return {
            "answer": "No relevant documents found.",
            "confidence": 0.0,
            "sources": [],
            "retrieval_time": retrieval_time
        }

    confidence = calculate_confidence(scores)   

    #If confidence is too low we avoid hallucinating
    if confidence < 0.3:

        return {
            "answer":"Insufficient information found in the documents.",
            "confidence":confidence,
            "sources":[],
            "retrieval_time":retrieval_time
        }

    #Adding source metadata
    #This will later help us display source file names and page numbers
    sources = get_sources(
        docs
    )

    #context
    #Combining all retrieved chunks into a single context string
    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    #prompt to llm to avoid hallucination
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

    #sending prompt to Gemini and measuring generation time
    start = time.time()

    try:

        response = llm.invoke(prompt)

    #Gemini daily quota issues can interrupt execution
    except Exception:

        return {
            "answer":"Gemini API quota exceeded. Please try again later.",
            "confidence":confidence,
            "sources":sources,
            "retrieval_time":retrieval_time
        }

    generation_time = float(time.time() - start)

    return {
        "answer":response.content,
        "confidence":confidence,
        "sources":sources,
        "retrieval_time":retrieval_time,
        "generation_time":generation_time
    }

if __name__ == "__main__":

    result = answer_question(
        "What is self attention?"
    )

    print("\nAnswer:\n")

    print(result["answer"])

    print("\nConfidence:")
    print(result["confidence"])

    print("\nSources:")

    for source in result["sources"]:

        print(
            source["file"],
            "Page",
            source["page"]
        )