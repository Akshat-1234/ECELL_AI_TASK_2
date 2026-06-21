# Langchain imports
from langchain_community.vectorstores import FAISS # for search
from langchain_community.embeddings import HuggingFaceEmbeddings
from pathlib import Path

# Importing documents created in preprocess.py
from preprocess import documents

BASE_DIR = Path(__file__).parent.parent

# Building the embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Building FAISS
# We convert the chunks into vectors and store them in a vector database
vectorstore = FAISS.from_documents(
    documents,
    embeddings
)

# Creating vectorstore folder to save vectors
vectorstore.save_local(
    str(BASE_DIR / "data" / "vectorstore")
)

print("Vectorstore saved successfully")

# Verification of working of the code using a normal question
# Here the answer is 5 nearest chunks of text
results = vectorstore.similarity_search(
    "Explain the self-attention mechanism in transformers",
    k=5
)

# The chunks are sliced to 300 words for efficient verification
for doc in results:

    print("\nSOURCE FILE:")
    print(doc.metadata["source_file"])

    print("PAGE:", doc.metadata["page_number"])

    print(doc.page_content[:300])