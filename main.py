# main.py
from retrieval.load_corpus import load_corpus
from retrieval.embed import embed_documents
from retrieval.search import build_faiss_index, search_index
from sentence_transformers import SentenceTransformer
from agent.reasoner import generate_answer


if __name__ == "__main__":
    print("loading corpus")
    corpus = load_corpus(limit=1000)
    print("embedding documents")
    embeddings = embed_documents(corpus)
    print("building faiss index")
    index = build_faiss_index(embeddings)

    print("loading sentence-transformer model")
    model = SentenceTransformer("Alibaba-NLP/gte-multilingual-base", trust_remote_code=True)

    while True:
        query = input("\n🔍 Enter your research question (or type 'exit'): ")
        if query.strip().lower() == "exit":
            break

        query_embedding = model.encode(query)
        idxs, _ = search_index(index, query_embedding, k=5)

        top_docs = [corpus[i]["text"] for i in idxs]
        answer = generate_answer(query, top_docs)

        print("\n🧠 Agent Answer:\n")
        print(answer)
        print("\n" + "="*80)
