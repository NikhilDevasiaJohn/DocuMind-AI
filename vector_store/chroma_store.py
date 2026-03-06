import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_chroma import Chroma
from embeddings.embedding_model import get_embedding_model


def create_vector_store(chunks):

    embeddings = get_embedding_model()

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="vector_db"
    )

    return vector_db


if __name__ == "__main__":

    from ingestion.document_loader import load_pdf
    from ingestion.text_splitter import split_documents

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pdf_path = os.path.join(project_root, "data", "test.pdf")

    docs = load_pdf(pdf_path)

    chunks = split_documents(docs)

    db = create_vector_store(chunks)

    print("Vector DB created successfully")