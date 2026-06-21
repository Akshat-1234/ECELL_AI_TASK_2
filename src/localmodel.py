# This file contains the local LLM pipeline
# We use Ollama to generate answers without external APIs

from langchain_community.llms import Ollama

# Loading local Llama model
llm = Ollama(
    model="llama3.2:3b"
)

def generate_local_answer(context, question):

    # Prompt to reduce hallucination
    prompt = f"""
Answer using ONLY the provided context.

If the answer is not present in the context,
say that the information is unavailable.

Use bullet points when necessary.

Context:
{context}

Question:
{question}
"""

    # Generating answer locally
    response = llm.invoke(
        prompt
    )

    return response