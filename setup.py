import os
from sentence_transformers import SentenceTransformer

def initialise_model():
    MODEL_PATH = "./models/all-MiniLM-L6-v2"

    if not os.path.exists(MODEL_PATH):
        print("Downloading embedding model...")
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        model.save(MODEL_PATH)
        print("Model installed")
    else:
        print("Embedding model already exists")
