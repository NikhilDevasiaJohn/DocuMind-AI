import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from pydantic import BaseModel

from rag_pipeline.rag_chain import create_rag_chain

app = FastAPI()

rag_chain = create_rag_chain()


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def root():
    return {"message": "DocuMind AI API running"}


@app.post("/ask")
def ask_question(request: QuestionRequest):

    response = rag_chain.invoke(request.question)

    return {
        "question": request.question,
        "answer": response
    }