# Pediatric Emergency Medicine Research Agent

This project implements a command-line research assistant for answering complex clinical questions using retrieval-augmented generation (RAG). The system performs dense retrieval over a domain-specific medical corpus and generates answers using a large language model (LLM).

## Project Purpose

This tool is designed to simulate a research assistant that can:
- Retrieve relevant scientific and clinical literature
- Reason over the retrieved documents to answer multi-hop questions
- Support clinicians or medical researchers in pediatric emergency medicine

The project showcases modular, domain-adapted RAG pipelines with strong retrieval and synthesis capabilities, intended for demonstration, research, or extension.

## Technology Stack

- Retrieval: FAISS + Sentence Transformers (e.g., `gte-multilingual-base`)
- LLM Reasoning: OpenAI `gpt-4o` or `gpt-3.5-turbo`
- Corpus: Scientific and clinical documents (e.g., from Hugging Face `scientific_papers`)
- Interface: Command-line application

## File Descriptions

| File/Directory          | Description |
|-------------------------|-------------|
| `main.py`               | Entry point that loads the corpus, builds the FAISS index, and runs the query interface. |
| `agent/embed.py`        | Encodes documents into dense vector embeddings using a sentence transformer. |
| `agent/load_corpus.py`  | Loads and optionally preprocesses the corpus for indexing. |
| `agent/search.py`       | Performs dense vector similarity search using FAISS. |
| `agent/reasoner.py`     | Sends the user query and retrieved documents to the OpenAI API for LLM-based reasoning. |
| `corpus/`               | (Optional) Folder for storing local or custom text corpora. |

## Example Usage

```
$ python main.py
Enter your research question: 
A 3-week-old infant presents with vomiting, hypotension, hyponatremia, hyperkalemia, and hypoglycemia. What is the likely diagnosis and initial management?

Agent Answer:
The presentation is consistent with congenital adrenal hyperplasia due to 21-hydroxylase deficiency...
```

## Setup Instructions

1. Clone the repository and create a virtual environment:

```bash
git clone https://github.com/your-username/research-agent.git
cd research-agent
conda create -n research_agent_env python=3.9
conda activate research_agent_env
pip install -r requirements.txt
```

2. Set your OpenAI API key:

```bash
export OPENAI_API_KEY=your-key-here
```

3. Run the agent:

```bash
python main.py
```

## Future Improvements

- Support hybrid retrieval (dense + sparse)
- Improve domain coverage with PubMed abstracts
- Add chunking for long documents
- Optional web UI using Streamlit or FastAPI

## Relevance

This project demonstrates practical experience with:
- Retrieval-augmented generation (RAG) systems
- Dense vector search with FAISS
- LLM prompt engineering and OpenAI integration
- Scientific and medical domain adaptation

It may be of interest to teams building AI agents for healthcare, question answering, or knowledge retrieval tasks.
