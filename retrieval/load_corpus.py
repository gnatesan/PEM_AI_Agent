# retrieval/load_corpus.py
from datasets import load_dataset

def load_corpus(split="train", limit=1000):
    dataset = load_dataset("pubmed_qa", "pqa_labeled", split=split)
    docs = dataset.select(range(min(limit, len(dataset))))
    return [{"id": str(i), "text": doc["context"]} for i, doc in enumerate(docs)]

