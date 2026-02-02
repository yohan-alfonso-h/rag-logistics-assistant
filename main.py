"""
RAG Logistics Assistant - Main Entry Point
CLI interface for the logistics RAG system.
"""

import argparse
import sys
import io
from pathlib import Path

# Fix Windows console encoding for emojis
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()


def cmd_download():
    """Download datasets from GitHub."""
    from rag.data_loader import download_datasets
    
    print("\n" + "=" * 50)
    print("[DESCARGA] DESCARGA DE DATASETS")
    print("=" * 50 + "\n")
    
    download_datasets()
    print("\n[OK] Descarga completada!")


def cmd_index():
    """Index documents into ChromaDB."""
    from rag.data_loader import load_documents
    from rag.vector_store import create_vector_store
    
    print("\n" + "=" * 50)
    print("[INDEX] INDEXACION DE DOCUMENTOS")
    print("=" * 50 + "\n")
    
    # Load documents
    documents = load_documents()
    
    if not documents:
        print("[ERROR] No hay documentos para indexar. Ejecuta primero: python main.py --download")
        return
    
    # Create vector store
    create_vector_store(documents)
    print("\n[OK] Indexacion completada!")


def cmd_query(question: str = None):
    """Query the RAG system."""
    from rag.vector_store import load_vector_store
    from rag.chain import create_rag_chain, query_rag, EXAMPLE_QUERIES
    
    print("\n" + "=" * 50)
    print("[RAG] RAG LOGISTICS ASSISTANT")
    print("=" * 50)
    
    # Load vector store
    store = load_vector_store()
    
    if store is None:
        print("\n[ERROR] No hay datos indexados.")
        print("   Ejecuta primero:")
        print("   1. python main.py --download")
        print("   2. python main.py --index")
        return
    
    # Create chain
    chain = create_rag_chain(store)
    
    if question:
        # Single query mode
        print(f"\n[?] Pregunta: {question}\n")
        response = query_rag(chain, question)
        print(f"[>] Respuesta:\n{response}\n")
    else:
        # Interactive mode
        print("\n[i] Modo interactivo. Escribe 'salir' para terminar.")
        print("    Escribe 'ejemplos' para ver preguntas de ejemplo.\n")
        
        while True:
            try:
                user_input = input("[?] Tu pregunta: ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() in ['salir', 'exit', 'quit', 'q']:
                    print("\n[!] Hasta luego!")
                    break
                    
                if user_input.lower() == 'ejemplos':
                    print("\n[EJEMPLOS] Preguntas de ejemplo:")
                    for i, q in enumerate(EXAMPLE_QUERIES, 1):
                        print(f"   {i}. {q}")
                    print()
                    continue
                
                # Query the RAG
                print("\n[...] Procesando...")
                response = query_rag(chain, user_input)
                print(f"\n[>] Respuesta:\n{response}\n")
                print("-" * 40 + "\n")
                
            except KeyboardInterrupt:
                print("\n\n[!] Hasta luego!")
                break


def cmd_demo():
    """Run a demo with example queries."""
    from rag.vector_store import load_vector_store
    from rag.chain import create_rag_chain, query_rag, EXAMPLE_QUERIES
    
    print("\n" + "=" * 50)
    print("[DEMO] RAG LOGISTICS ASSISTANT")
    print("=" * 50)
    
    # Load vector store
    store = load_vector_store()
    
    if store is None:
        print("\n[ERROR] No hay datos indexados. Ejecuta setup primero:")
        print("   python main.py --download")
        print("   python main.py --index")
        return
    
    # Create chain
    chain = create_rag_chain(store)
    
    # Run demo queries
    demo_queries = EXAMPLE_QUERIES[:3]  # First 3 examples
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n{'='*50}")
        print(f"[Demo {i}/{len(demo_queries)}]")
        print(f"{'='*50}")
        print(f"\n[?] Pregunta: {query}\n")
        
        response = query_rag(chain, query)
        print(f"[>] Respuesta:\n{response}\n")
        
        if i < len(demo_queries):
            input("Presiona Enter para continuar...")


def main():
    parser = argparse.ArgumentParser(
        description="RAG Logistics Assistant - Sistema de preguntas y respuestas sobre logistica",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python main.py --download          # Descarga los datasets
  python main.py --index             # Indexa los documentos
  python main.py --query "pregunta"  # Hace una pregunta especifica
  python main.py --interactive       # Modo interactivo
  python main.py --demo              # Demo con ejemplos
        """
    )
    
    parser.add_argument(
        '--download', 
        action='store_true',
        help='Descarga los datasets de logistica desde GitHub'
    )
    
    parser.add_argument(
        '--index', 
        action='store_true',
        help='Indexa los documentos en ChromaDB'
    )
    
    parser.add_argument(
        '--query', '-q',
        type=str,
        help='Hace una pregunta especifica al sistema'
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Inicia el modo interactivo'
    )
    
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Ejecuta una demostracion con preguntas de ejemplo'
    )
    
    args = parser.parse_args()
    
    # Execute based on arguments
    if args.download:
        cmd_download()
    elif args.index:
        cmd_index()
    elif args.query:
        cmd_query(args.query)
    elif args.interactive:
        cmd_query()
    elif args.demo:
        cmd_demo()
    else:
        # Default: show help
        parser.print_help()
        print("\n[TIP] Quickstart:")
        print("   1. Configura tu .env con OPENAI_API_KEY")
        print("   2. python main.py --download")
        print("   3. python main.py --index")
        print("   4. python main.py --interactive")


if __name__ == "__main__":
    main()
