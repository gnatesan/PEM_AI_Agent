"""
LangChain BaseTool: VectorSearchTool
Performs semantic retrieval over the unified pediatric emergency medicine corpus.
"""

from typing import Optional
from langchain.tools import BaseTool
from langchain_community.embeddings import HuggingFaceEmbeddings
import faiss
import pickle
import os
import numpy as np


class VectorSearchTool(BaseTool):
    name: str = "vector_search"
    description: str = (
        "Searches a pediatric emergency medicine corpus for relevant documents "
        "based on a natural language query using FAISS similarity search."
    )

    index_path: str = "scripts/data/expanded_faiss_index.faiss"
    corpus_path: str = "scripts/data/expanded_corpus.pkl"
    embeddings: Optional[HuggingFaceEmbeddings] = None
    index: Optional[faiss.Index] = None
    corpus: Optional[list] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="ibm-granite/granite-embedding-125m-english"
        )
        self.load_resources()

    def load_resources(self):
        """Load FAISS index and corpus into memory."""
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Missing FAISS index: {self.index_path}")
        if not os.path.exists(self.corpus_path):
            raise FileNotFoundError(f"Missing corpus pickle: {self.corpus_path}")

        print("Loading FAISS index...")
        self.index = faiss.read_index(self.index_path)

        print("Loading corpus...")
        with open(self.corpus_path, "rb") as f:
            self.corpus = pickle.load(f)

    def _run(self, query: str, k: int = 5, run_manager: Optional[object] = None) -> str:
        """Synchronously search the FAISS index for the most relevant documents."""
        if self.index is None or self.corpus is None:
            self.load_resources()

        query_vector = self.embeddings.embed_query(query)
        #scores, indices = self.index.search([query_vector], k)
        scores, indices = self.index.search(np.array([query_vector]), k)        

        results = []
        for i in indices[0]:
            if i in self.corpus:
                results.append(self.corpus[i])
            else:
                results.append(f"[Missing chunk for index {i}]")

        return "\n\n---\n\n".join(results)

    async def _arun(self, query: str, k: int = 5, run_manager: Optional[object] = None) -> str:
        """Async version (not used by default but required by BaseTool)."""
        return self._run(query, k)

