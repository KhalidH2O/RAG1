import os
from sentence_transformers import SentenceTransformer

EMB_MODEL = "BAAI/bge-small-en-v1.5"
# MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMB_MODEL_PATH = "./models/bge-small-en-v1.5"

def initialise_model():
    if not os.path.exists(EMB_MODEL_PATH):
        print("Downloading embedding model...")
        model = SentenceTransformer(EMB_MODEL,trust_remote_code=True)
        model.save(EMB_MODEL_PATH)
        print("Model installed")
    else:
        print("Embedding model already exists")