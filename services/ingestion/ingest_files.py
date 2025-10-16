# services/ingestion/ingest_faq.py
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def load_doc(path):
    if path.endswith(".pdf"):
        loader = PyPDFLoader(path)
    else:
        loader = TextLoader(path, encoding="utf-8")
    return loader.load()

def ingest(paths, index_path="faiss_faq_index"):
    docs = []
    for p in paths:
        docs += load_doc(p)
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(docs)

    embed = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vect = FAISS.from_documents(chunks, embed)
    vect.save_local(index_path)
    print("Saved FAISS index to", index_path)

if __name__ == "__main__":
    # Example: python ingest_faq.py ./data/faq.txt
    import sys
    ingest(sys.argv[1:])