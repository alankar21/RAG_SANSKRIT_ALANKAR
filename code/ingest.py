import os
import shutil

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(BASE_DIR, "code", "faiss_index")


def main():
    # ✅ Remove old FAISS index if exists
    if os.path.exists(DB_PATH):
        print("🗑️ Removing old FAISS index...")
        shutil.rmtree(DB_PATH)

    print("📥 Loading Sanskrit documents from /data ...")

    loader = DirectoryLoader(
        DATA_DIR,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )

    docs = loader.load()

    if len(docs) == 0:
        raise FileNotFoundError(f"No .txt files found inside: {DATA_DIR}")

    print(f"✅ Loaded documents: {len(docs)}")

    print("✂️ Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    print(f"✅ Total chunks: {len(chunks)}")

    print("🧠 Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    print("📦 Building FAISS index...")
    db = FAISS.from_documents(chunks, embeddings)

    os.makedirs(DB_PATH, exist_ok=True)
    db.save_local(DB_PATH)

    print(f"✅ Saved index to: {DB_PATH}")


if __name__ == "__main__":
    main()
