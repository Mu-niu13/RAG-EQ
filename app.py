from fastapi import FastAPI
from pydantic import BaseModel
from rag_engine import rag_pipeline
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
