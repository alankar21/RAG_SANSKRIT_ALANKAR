# Sanskrit Document Retrieval-Augmented Generation (RAG) System (CPU Only)

## Project Title
**Sanskrit Document Retrieval-Augmented Generation (RAG) System**

This project implements an end-to-end **Retrieval-Augmented Generation (RAG)** pipeline for Sanskrit documents.  
It ingests Sanskrit `.txt` files, builds a FAISS vector index, retrieves relevant context, and generates answers using a CPU-based LLM.

---

## Folder Structure
RAG_Sanskrit_Alankar/
├── code/
│ ├── app.py
│ ├── ingest.py
│ ├── requirements.txt
│ └── faiss_index/ # Created after running ingest.py
├── data/
│ ├── murkhabhritya.txt
│ ├── devbhakta.txt
│ ├── ghantakarna.txt
│ ├── kalidasa.txt
│ └── sheetam.txt
├── report/
│ ├── report.md
│ └── report.pdf
└── README.md

---

## Features
- ✅ CPU-only inference (No GPU required)
- ✅ Sanskrit document ingestion (`.txt`)
- ✅ Preprocessing + chunking
- ✅ Semantic retrieval using FAISS vector store
- ✅ Answer generation using CPU-based LLM (`google/flan-t5-small`)
- ✅ Interactive query interface (Terminal / CLI)

---

## Tech Stack
- **Python 3.10+**
- **LangChain**
- **FAISS**
- **SentenceTransformers**
- **HuggingFace Transformers**
- **LLM Model:** `google/flan-t5-small` (CPU-friendly)

---

## Setup Instructions (VS Code + Terminal)

### 1) Create Virtual Environment

```powershell
python -m venv venv
.\venv\Scripts\activate


Install Dependencies
pip install -r code/requirements.txt

Execution Steps
Step 1: Build Vector Index (Ingestion)

This loads documents from /data, splits them into chunks, generates embeddings, and builds a FAISS vector index.

python code/ingest.py  

📥 Loading Sanskrit documents from /data ...
✅ Loaded documents: 5
✂️ Splitting into chunks...
✅ Total chunks: XX
🧠 Creating embeddings...
📦 Building FAISS index...
✅ Saved index to: code/faiss_index


Step 2: Run Query Interface (RAG QA System)
python code/app.py

✅ Sanskrit RAG System Ready (LangChain + FAISS + FLAN-T5 CPU)
🟡 Query (type 'exit' to stop):