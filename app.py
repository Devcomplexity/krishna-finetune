from fastapi import FastAPI
from pydantic import BaseModel
from rag_engine import generate_response

app = FastAPI()

class Question(BaseModel):
    query: str

@app.post("/ask")
def ask_question(q: Question):
    answer = generate_response(q.query)
    return {"answer": answer}