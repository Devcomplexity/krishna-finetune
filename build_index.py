import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load verses
with open("data/gita_verses.json", "r", encoding="utf-8") as f:
    verses = json.load(f)

texts = [v["text"] for v in verses]

print("Generating embeddings...")
embeddings = embedder.encode(texts)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Save index and metadata
faiss.write_index(index, "gita.index")

with open("verses.pkl", "wb") as f:
    pickle.dump(verses, f)

print("Index built successfully.")