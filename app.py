from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from rag import RAGService
#from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI()
rag = RAGService()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

app.mount("/static", StaticFiles(directory="static"), name= "static")

class QueryRequest(BaseModel):
    question: str

# @app.get("/")
# def home():
#     return {"message": "RAG API is running."}

@app.get("/")
def serveui():
    return FileResponse("static/index.html")

@app.post("/query")
def query_rag(req: QueryRequest):
    answer, context = rag.get_answer(req.question)

    return{
        "question": req.question,
        "answer" : answer,
        "context": context
    }

@app.post("/upload")
def upload_file(file: UploadFile = File(...)):

    file_bytes = file.file.read()

    result = rag.add_file(file_bytes, file.filename)

    return {"message": result}