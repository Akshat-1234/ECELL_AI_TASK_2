from fastapi import FastAPI
from pydantic import BaseModel

# Importing the main RAG pipeline
from src.train import answer_question

# Creating FastAPI application
app = FastAPI()

# Input schema for user queries
# User sends question in json format
class QueryRequest(BaseModel):
    query : str


# Home endpoint
# Used to check whether server is running properly
@app.get("/")
def home():

    return {
        "message":"RAG API is running successfully"
    }


# Main endpoint for question answering
# User sends a query and receives answer, confidence and sources
@app.post("/query")
def query(request: QueryRequest):

    # Passing user question to the RAG pipeline
    result = answer_question(
        request.query
    )

    # Returning everything in json format
    return {
        "answer": result["answer"],
        "confidence": float(result["confidence"]),
        "sources": result["sources"],
        "retrieval_time": round(
            float(result["retrieval_time"]),
            4
        ),
        "generation_time": round(
            float(
                result.get(
                    "generation_time",
                    0
                )
            ),
            4
        )
    }

# For local testing
if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )