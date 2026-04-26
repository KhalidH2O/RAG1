from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from fastapi import FastAPI

app = FastAPI()

emb = HuggingFaceEmbeddings()
print("Setup works")

loader = TextLoader("data.txt")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size = 200,chunk_overlap = 50)

chunks = splitter.split_documents(documents)

print("Chunks:")
for chunk in chunks:
    print(chunk.page_content)
    print("--------")

db = Chroma.from_documents(chunks,emb)

print("Vector DB created successfully")

# query = "What is the point of minecraft?"

llm = OllamaLLM(model = "llama3.2")

@app.post("/query")
def query_rag(query: str):
    results = db.similarity_search(query)
    
    context = "\n\n".join([r.page_content for r in results])
                       
    prompt = f"""
    You must answer using ONLY the context below.
    If the answer is not in the context, say "I don't know"

    Context:
    {context}

    Question:
    {query}
    """

    response = llm.invoke(prompt)

    return {
        "question" : query,
        "answer" : response,
        "context" : context
    }