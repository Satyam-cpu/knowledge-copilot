import os
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    DirectoryLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def load_documents(data_dir="data/docs"):
    """Saare documents load karo"""
    documents = []
    
    # Text files
    for file in os.listdir(data_dir):
        filepath = os.path.join(data_dir, file)
        
        if file.endswith(".txt"):
            print(f"  📄 Loading: {file}")
            loader = TextLoader(filepath, encoding="utf-8")
            documents.extend(loader.load())
            
        elif file.endswith(".pdf"):
            print(f"  📕 Loading: {file}")
            loader = PyPDFLoader(filepath)
            documents.extend(loader.load())
    
    return documents

def chunk_documents(documents):
    """Documents ko chunks mein todo"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    return chunks

def create_vectordb(chunks):
    """Vector DB banao aur save karo"""
    print("\n  🔢 Embeddings ban rahi hain...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    
    print("  💾 ChromaDB mein store ho raha hai...")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    return vectordb

def ingest():
    """Main function — sab kuch run karo"""
    print("="*50)
    print("📚 DOCUMENT INGESTION SHURU")
    print("="*50)
    
    print("\n📂 Documents load ho rahe hain...")
    documents = load_documents()
    print(f"✅ Total documents: {len(documents)}")
    
    print("\n✂️  Chunks ban rahe hain...")
    chunks = chunk_documents(documents)
    print(f"✅ Total chunks: {len(chunks)}")
    
    print("\n🗄️  Vector DB ban raha hai...")
    vectordb = create_vectordb(chunks)
    print(f"✅ Vector DB ready!")
    
    print("\n" + "="*50)
    print("🎉 INGESTION COMPLETE!")
    print(f"   Documents: {len(documents)}")
    print(f"   Chunks: {len(chunks)}")
    print("="*50)
    
    return vectordb

if __name__ == "__main__":
    ingest()
