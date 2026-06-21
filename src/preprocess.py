import fitz  # pdf reader
import os    # to manage directories

# Langchain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter # to make chunks
from langchain_core.documents import Document
from pathlib import Path

# Paper Extraction from folder
documents = []

BASE_DIR = Path(__file__).parent.parent

# Here we first use os module to find a given file in the pdfs folder,
# Then we extract text page by page
# We split each page into chunks and save metadata with every chunk
# We save file name and page number because we are going to need to cite
# the exact source while answering
pdf_dir = BASE_DIR / "data" / "pdfs"

# Now the text is split into chunks if size 1000
# We use chunk overlap to divide text in a way that 2 adjacent segments contain common parts
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
# Here we read every pdf present inside the folder

for file in os.listdir(pdf_dir):

    if file.endswith(".pdf"):

        path = pdf_dir / file
        pdf = fitz.open(path)

        # We go page by page so that page number information is preserved
        for page_num, page in enumerate(pdf):

            text = page.get_text()

            # Split current page into chunks
            split_chunks = splitter.split_text(text)

            # Now we convert each chunk into Langchain documents
            # Metadata is stored because it will be useful for source citation
            for chunk_id, chunk in enumerate(split_chunks):

                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source_file": file,
                            "page_number": page_num + 1,
                            "chunk_id": chunk_id
                        }
                    )
                )

print(f"Total chunks created : {len(documents)}")