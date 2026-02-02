"""
Vector Store Module
Manages ChromaDB vector database for document storage and retrieval.
"""

import os
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


def get_chroma_dir() -> Path:
    """Get the ChromaDB storage directory."""
    base_dir = Path(__file__).parent.parent
    chroma_dir = base_dir / "chroma_db"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    return chroma_dir


def get_embeddings() -> OpenAIEmbeddings:
    """
    Create OpenAI embeddings instance.
    Uses text-embedding-ada-002 by default.
    """
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        # Uses OPENAI_API_KEY from environment
    )


def create_vector_store(
    documents: List[Document],
    collection_name: str = "logistics_docs"
) -> Chroma:
    """
    Create a new ChromaDB vector store from documents.
    
    Args:
        documents: List of LangChain Documents to embed
        collection_name: Name for the Chroma collection
        
    Returns:
        Chroma vector store instance
    """
    print(f"[INDEX] Creando vector store con {len(documents)} documentos...")
    
    embeddings = get_embeddings()
    chroma_dir = get_chroma_dir()
    
    # Create vector store with documents
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(chroma_dir)
    )
    
    print(f"[OK] Vector store creado y guardado en {chroma_dir}")
    return vector_store


def load_vector_store(collection_name: str = "logistics_docs") -> Optional[Chroma]:
    """
    Load an existing ChromaDB vector store.
    
    Args:
        collection_name: Name of the Chroma collection
        
    Returns:
        Chroma vector store instance, or None if not found
    """
    chroma_dir = get_chroma_dir()
    
    if not (chroma_dir / "chroma.sqlite3").exists():
        print("[WARN] No se encontro vector store existente")
        return None
    
    print(f"[LOAD] Cargando vector store desde {chroma_dir}...")
    
    embeddings = get_embeddings()
    
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(chroma_dir)
    )
    
    # Check if collection has documents
    collection = vector_store._collection
    count = collection.count()
    print(f"[OK] Vector store cargado con {count} documentos")
    
    return vector_store


def similarity_search(
    vector_store: Chroma,
    query: str,
    k: int = 4
) -> List[Document]:
    """
    Search for similar documents.
    
    Args:
        vector_store: Chroma vector store instance
        query: Search query
        k: Number of results to return
        
    Returns:
        List of similar documents
    """
    results = vector_store.similarity_search(query, k=k)
    return results


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    # Try to load existing store
    store = load_vector_store()
    
    if store:
        # Test search
        query = "envíos con retrasos"
        print(f"\n[SEARCH] Buscando: '{query}'")
        results = similarity_search(store, query, k=2)
        
        for i, doc in enumerate(results, 1):
            print(f"\n--- Resultado {i} ---")
            print(doc.page_content[:300])
    else:
        print("\n[TIP] Ejecuta primero: python -m rag.data_loader")
        print("      Luego indexa con: python main.py --index")
