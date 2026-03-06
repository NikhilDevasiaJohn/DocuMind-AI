import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import OpenAIEmbeddings
from config.settings import OPENAI_API_KEY



def get_embedding_model():

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY
    )

    return embeddings


if __name__ == "__main__":

    embedding_model = get_embedding_model()

    text = "This is a test sentence"

    vector = embedding_model.embed_query(text)

    print("Embedding vector length:", len(vector))