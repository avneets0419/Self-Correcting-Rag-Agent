from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
import os

def clean_text(text: str) -> str:
    # Fix word-by-word newlines from bad PDF parsing
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    # Collapse multiple spaces
    text = re.sub(r' +', ' ', text)
    return text.strip()

def build_retriever(data_dir="data/"):
    index_path = "faiss_index"
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if os.path.exists(index_path):
        # Load existing index — no re-embedding needed
        vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    else:
        # Build fresh and save
        loader = PyPDFDirectoryLoader(data_dir)
        docs = loader.load()
        for doc in docs:
            doc.page_content = clean_text(doc.page_content)
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(index_path)

    return vectorstore.as_retriever(search_kwargs={"k": 4})