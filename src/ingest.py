from unittest import loader
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
from langchain_chroma import Chroma

load_dotenv()
DB_NAME = str(Path(__file__).parent.parent / "db" / "mani_resume_db")
openai_api_key = os.getenv("OPENAI_API_KEY")

# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
def load_documents():
    print("Loading documents...")
    loader = PDFPlumberLoader(str(Path(__file__).parent.parent / "docs" / "Mani-Resume.pdf"))
    data = loader.load()
    print("data", [data.page_content for data in data])
    print(f"Loaded {len(data)} documents.")
    return data

def split_documents(data): 
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
    texts = text_splitter.split_documents(data)
    print("spits", [text.page_content for text in texts])
    print(f"Split into {len(texts)} chunks.")
    return texts

def create_embeddings(chunks):
    print("Creating embeddings...")
    if os.path.exists(DB_NAME):
        Chroma(persist_directory=DB_NAME, embedding_function=embeddings).delete_collection()
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=DB_NAME)
    collection = vectorstore._collection
    count = collection.count()
    sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    dimension = len(sample_embedding)
    print(f"There are {count} vectors with dimension {dimension} in the vectorstore.")
    return vectorstore

if __name__ == "__main__":
    data = load_documents()
    chunks = split_documents(data)
    create_embeddings(chunks)



