from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM

import os
import hashlib

class RAGService:
    def __init__(self):
        self.emb = HuggingFaceEmbeddings()

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size = 200,
            chunk_overlap = 50
        )

        if os.path.exists("./chroma_db"):
            print("Loading existing DB...")
            self.db = Chroma(
                persist_directory= "./chroma_db",
                embedding_function= self.emb
            )
        else:
            print("Creating a new DB...")
            self.loader = TextLoader("data.txt")
            self.documents = self.loader.load()   

            self.chunks = self.splitter.split_documents(self.documents)
            
            self.db = Chroma.from_documents(
                self.chunks,
                self.emb,
                persist_directory='./chroma_db'
            )

        self.llm = OllamaLLM(model='llama3.2')

    def get_answer(self,query: str):
        results = self.db.similarity_search(query)

        context = "\n\n".join([r.page_content for r in results])

        prompt = f"""
        You MUST answer using ONLY the context below.
        If the answer is not in the context, say "Found no relevant data!"
        
        Context:
        {context}

        Query:
        {query}
        """

        response = self.llm.invoke(prompt)
        
        return response, context

    def add_file(self, file_bytes, filename):
        from langchain_core.documents import Document

        file_hash = hashlib.md5(file_bytes).hexdigest()

        existing = self.db.get(where = {"file_hash": file_hash})
        if existing["ids"]:
            return "File already uploaded"
        
        try:
            text = file_bytes.decode("utf-8")
        except:
            return "Unsupported file format(Only .txt allowed)"
        
        doc = Document(
            page_content= text,
            metadata={
                "source": filename,
                "file_hash": file_hash
            }
        )

        chunks = self.splitter.split_documents([doc])

        for chunk in chunks:
            chunk.metadata["source"] = filename
            chunk.metadata["file_hash"] = file_hash

        self.db.add_documents(chunks)

        return "File uploaded successfully"
    
        