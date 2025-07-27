# main.py
import pickle
import faiss
from retrieval.search import build_faiss_index, search_index
from sentence_transformers import SentenceTransformer
from agent.reasoner import generate_answer

def load_corpus_and_index(corpus_path="data/expanded_corpus.pkl", index_path="data/expanded_faiss_index.faiss"):
    print("Loading saved corpus and FAISS index...")
    with open(corpus_path, "rb") as f:
        corpus = pickle.load(f)
    index = faiss.read_index(index_path)
    return corpus, index

if __name__ == "__main__":
     # Load corpus and FAISS index
    corpus, index = load_corpus_and_index()    

    print("loading sentence-transformer model")
    model = SentenceTransformer("Alibaba-NLP/gte-multilingual-base", trust_remote_code=True)

    while True:
        query = input("\nEnter your research question (or type 'exit'): ")
        if query.strip().lower() == "exit":
            break

        query_embedding = model.encode(query)
        idxs, _ = search_index(index, query_embedding, k=5)

        top_docs = [corpus[i] for i in idxs]
        answer = generate_answer(query, top_docs)

        print("\nAgent Answer:\n")
        print(answer)
        print("\n" + "="*80)
