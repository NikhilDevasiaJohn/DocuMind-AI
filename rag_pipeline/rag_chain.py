import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from embeddings.embedding_model import get_embedding_model
from config.settings import OPENAI_API_KEY


def load_vector_db():

    embeddings = get_embedding_model()

    db = Chroma(
        persist_directory="vector_db",
        embedding_function=embeddings
    )

    return db


def create_rag_chain():

    db = load_vector_db()

    retriever = db.as_retriever()

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    )

    prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:

{context}

Question: {question}
""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


if __name__ == "__main__":

    chain = create_rag_chain()

    while True:

        query = input("\nAsk a question: ")

        if query.lower() == "exit":
            break

        response = chain.invoke(query)

        print("\nAnswer:", response)