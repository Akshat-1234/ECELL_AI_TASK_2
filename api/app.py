from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Literal

# Importing all three RAG pipelines
from src.train import (
    answer_question,
    answer_question_rerank,
    answer_question_local,
)

# Creating FastAPI application
app = FastAPI()

# Maps the "pipeline" field in the request to the actual function to call.
# Keeping this as a simple dict means adding a 4th pipeline later is a
# one-line change here, nothing else in this file needs to move.
PIPELINES = {
    "api": answer_question,
    "rerank": answer_question_rerank,
    "local": answer_question_local,
}


# Input schema for user queries
# User sends question in json format, with an optional pipeline choice.
# Defaults to "api" so any existing client that doesn't send a
# "pipeline" field at all still gets the original behavior.
class QueryRequest(BaseModel):
    query: str
    pipeline: Optional[Literal["api", "rerank", "local"]] = "api"


# Home endpoint
# Used to check whether server is running properly
@app.get("/")
def home():

    return {
        "message": "RAG API is running successfully",
        "available_pipelines": list(PIPELINES.keys()),
    }


# Main endpoint for question answering
# User sends a query (and optionally picks a pipeline) and receives
# answer, confidence and sources.
@app.post("/query")
def query(request: QueryRequest):

    pipeline_fn = PIPELINES[request.pipeline]

    # Passing user question to the selected RAG pipeline
    result = pipeline_fn(
        request.query
    )

    # Returning everything in json format.
    # All three pipelines now return the same shape (see train.py),
    # so this response building stays identical regardless of which
    # pipeline was used — nothing pipeline-specific needed here.
    return {
        "pipeline": request.pipeline,
        "answer": result["answer"],
        "confidence": float(result.get("confidence", 0.0)),
        "sources": result["sources"],
        "retrieval_time": round(
            float(result.get("retrieval_time", 0.0)),
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