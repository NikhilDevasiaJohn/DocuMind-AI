import os
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    return chunks


if __name__ == "__main__":

    from document_loader import load_pdf

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pdf_path = os.path.join(project_root, "data", "test.pdf")
    docs = load_pdf(pdf_path)

    chunks = split_documents(docs)

    print("Total chunks:", len(chunks))
    print("\nFirst chunk preview:\n")
    print(chunks[0].page_content[:500])