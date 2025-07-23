# retrieval/search.py
import faiss
import numpy as np

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def search_index(index, query_embedding, k=5):
    D, I = index.search(np.array([query_embedding]), k)
    return I[0], D[0]

