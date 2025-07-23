# retrieval/embed.py
from sentence_transformers import SentenceTransformer
import numpy as np

def embed_documents(documents, model_name="ibm-granite/granite-embedding-125m-english"):
    model = SentenceTransformer(model_name)
    texts = [doc["text"] for doc in documents]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings

