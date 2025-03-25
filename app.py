from fastapi import FastAPI
from pydantic import BaseModel
from rag_engine import rag_pipeline

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/rag/")
async def rag_endpoint(query: Query):
    print(f"[API] Received query: {query.question}")
    answer = rag_pipeline(query.question)
    return {"question": query.question, "answer": answer}

@app.get("/")
def root():
    return {"message": "Welcome to the EQ RAG API!"}
