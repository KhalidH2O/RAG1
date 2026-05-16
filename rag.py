import os
import hashlib

# os.environ["TRANSFORMERS_OFFLINE"] = "1"
# os.environ["HF_HUB_OFFLINE"] = "1"

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
import emb_setup

class RAGService:
    def __init__(self):
        self.MODEL = "llama3.2"

        emb_setup.initialise_model()
        self.emb = HuggingFaceEmbeddings(model_name = emb_setup.EMB_MODEL_PATH, encode_kwargs = {"normalize_embeddings": True})

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size = 600,
            chunk_overlap = 120
        )

        if os.path.exists("./chroma_db"):
            print("Loading existing DB...")
            self.db = Chroma(
                persist_directory= "./chroma_db",
                embedding_function= self.emb
            )
        else:
            print("Creating a new DB...")
            self.loader = TextLoader("./data/data.txt")
            self.documents = self.loader.load()   

            self.chunks = self.splitter.split_documents(self.documents)
            
            self.db = Chroma.from_documents(
                self.chunks,
                self.emb,
                persist_directory='./chroma_db'
            )
        
        self.llm = OllamaLLM(model=self.MODEL,temperature=0.1)

    def get_answer(self,query: str):
        generated_questions = self.multi_query(query)

        queries = generated_questions.strip().split("\n")

        all_results = []
        for q in queries:
            res = self.db.similarity_search(q, k=2)
            all_results.extend(res)

        unique_docs = {r.page_content: r for r in all_results}.values()
    
        context = "\n\n".join([r.page_content for r in unique_docs])
        
        print("\nAll Retrieved Documents:\n--------------------\n", all_results,"\n\n")
        print("\nUnique Documents:\n---------------------\n", unique_docs,"\n\n")
        print("\nContext\n---------\n",context,"\n\n")
        
        prompt = f"""
        You MUST answer using the context below.
        If the answer is not in the context, say "Found no relevant data!"
        Provide details if relevant.

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
    
    def multi_query(self,query: str):
        multi_prompt = f"""
        Generate 4 different rephrasings of the following question. 
        Each variation must be concise, semantically equivalent, and use different wording or perspective.
        No introductory lines.

        Question: "{query}"
        """
        
        generated_questions = self.llm.invoke(multi_prompt)
        print(generated_questions)
        
        return generated_questions