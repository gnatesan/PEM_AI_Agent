# scripts/pdf_ingest.py

import os
import pickle
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from tqdm import tqdm

# Parameters
PDF_PATH = "scripts/PEM-Guide-7.0-May-2020.pdf"
CHUNK_SIZE = 512
EMBEDDING_MODEL_NAME = "Alibaba-NLP/gte-multilingual-base"
DATA_DIR = "data"

# Step 1: Read PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = "\n".join([page.extract_text() or "" for page in reader.pages])
    return text

# Step 2: Split text into chunks
def split_text(text, chunk_size=CHUNK_SIZE):
    paragraphs = text.split("\n")
    chunks = []
    buffer = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(buffer) + len(para) <= chunk_size:
            buffer += " " + para
        else:
            chunks.append(buffer.strip())
            buffer = para
    if buffer:
        chunks.append(buffer.strip())
    return chunks

def embed_chunks(chunks, model_name, batch_size=2):
    model = SentenceTransformer(model_name, trust_remote_code=True)
    embeddings = model.encode(
        chunks,
        batch_size=batch_size,
        max_length=512,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    return embeddings

# Step 4: Build FAISS index
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# Step 5: Save index and corpus
def save_outputs(index, corpus, data_dir):
    Path(data_dir).mkdir(exist_ok=True)
    faiss.write_index(index, os.path.join(data_dir, "expanded_faiss_index.faiss"))
    with open(os.path.join(data_dir, "expanded_corpus.pkl"), "wb") as f:
        pickle.dump(corpus, f)

def main():
    print("📄 Extracting text from PDF...")
    text = extract_text_from_pdf(PDF_PATH)

    print("✂️ Splitting text into chunks...")
    chunks = split_text(text)

    print("🔢 Embedding chunks...")
    embeddings = embed_chunks(chunks, EMBEDDING_MODEL_NAME)

    print("📦 Building FAISS index...")
    index = build_faiss_index(embeddings)

    print("💾 Saving index and corpus to disk...")
    save_outputs(index, chunks, DATA_DIR)

    print(f"✅ Ingestion complete. {len(chunks)} chunks processed.")

if __name__ == "__main__":
    main()

