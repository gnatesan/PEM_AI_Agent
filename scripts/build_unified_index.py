import os
import pickle
from pathlib import Path

from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from datasets import load_dataset
import faiss
from tqdm import tqdm

# Constants
PDF_FILES = [
    "PEM-Guide-7.0-May-2020.pdf",
    "WHO_Peds_EM_Triage.pdf",
    "Patient_Safety_in_the_Pediatric_Emergency_Care_Setting_v2018_1.pdf"
]
EMBEDDING_MODEL_NAME = "Alibaba-NLP/gte-multilingual-base"
CHUNK_SIZE = 512
EMBED_BATCH_SIZE = 2
OUTPUT_INDEX_PATH = "data/expanded_faiss_index.faiss"
OUTPUT_CORPUS_PATH = "data/expanded_corpus.pkl"


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text


def chunk_text(text, chunk_size=CHUNK_SIZE):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


def load_pdf_documents():
    print("Loading and chunking PDFs...")
    corpus = []
    doc_id = 0
    for pdf_path in PDF_FILES:
        text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(text)
        for chunk in chunks:
            corpus.append({"id": str(doc_id), "text": chunk})
            doc_id += 1
    return corpus


def load_pubmed_documents(limit=1000):
    print("Loading PubMed QA dataset...")
    dataset = load_dataset("pubmed_qa", "pqa_labeled", split="train")
    docs = dataset.select(range(min(limit, len(dataset))))
    return [{"id": f"pubmed_{i}", "text": doc["context"]} for i, doc in enumerate(docs)]


def embed_corpus(corpus, model_name):
    print("Embedding all documents...")
    model = SentenceTransformer(model_name, trust_remote_code=True)
    texts = [doc["text"] for doc in corpus]
    embeddings = model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        max_length=512
    )
    return embeddings


def save_index_and_corpus(embeddings, corpus, index_path, corpus_path):
    print("Saving index and corpus to disk...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, index_path)
    with open(corpus_path, "wb") as f:
        pickle.dump(corpus, f)


def main():
    Path("data").mkdir(parents=True, exist_ok=True)
    pdf_corpus = load_pdf_documents()
    pubmed_corpus = load_pubmed_documents()
    full_corpus = pdf_corpus + pubmed_corpus
    embeddings = embed_corpus(full_corpus, EMBEDDING_MODEL_NAME)
    save_index_and_corpus(embeddings, full_corpus, OUTPUT_INDEX_PATH, OUTPUT_CORPUS_PATH)
    print(f"Ingestion complete. {len(full_corpus)} documents processed.")


if __name__ == "__main__":
    main()

