"""
RAG Chain Module
Implements the Retrieval Augmented Generation pipeline using LangChain and OpenAI.
"""

from typing import Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma


# Spanish-focused prompt template for logistics queries
LOGISTICS_PROMPT = """Eres un asistente experto en logística y cadena de suministro. 
Tu trabajo es responder preguntas basándote ÚNICAMENTE en el contexto proporcionado.

Contexto de datos de logística:
{context}

Pregunta del usuario: {question}

Instrucciones:
1. Responde en español de manera clara y profesional
2. Usa SOLO la información del contexto proporcionado
3. Si la información no está en el contexto, indica que no tienes datos suficientes
4. Cuando menciones números o estadísticas, cita la fuente (orden, carrier, etc.)
5. Organiza tu respuesta de manera estructurada si es apropiado

Respuesta:"""


def format_docs(docs) -> str:
    """Format retrieved documents into a single context string."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get('source', 'unknown')
        formatted.append(f"[Documento {i} - {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def create_rag_chain(
    vector_store: Chroma,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.3,
    k: int = 4
) -> Any:
    """
    Create a RAG chain for logistics queries.
    
    Args:
        vector_store: ChromaDB vector store with indexed documents
        model_name: OpenAI model to use for generation
        temperature: LLM temperature (lower = more focused)
        k: Number of documents to retrieve
        
    Returns:
        LangChain runnable chain
    """
    # Create retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    
    # Create LLM
    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature
    )
    
    # Create prompt
    prompt = ChatPromptTemplate.from_template(LOGISTICS_PROMPT)
    
    # Build the chain
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain


def query_rag(
    chain: Any,
    question: str,
    verbose: bool = False
) -> str:
    """
    Query the RAG system.
    
    Args:
        chain: The RAG chain
        question: User's question
        verbose: If True, print debug info
        
    Returns:
        Generated response string
    """
    if verbose:
        print(f"[SEARCH] Procesando: {question}")
    
    response = chain.invoke(question)
    
    return response


# Predefined example queries for testing
EXAMPLE_QUERIES = [
    "¿Cuáles son los principales modos de envío utilizados?",
    "¿Qué carriers manejan las tarifas más bajas?",
    "Describe los problemas de entrega más comunes",
    "¿Cuáles son las rutas de envío más utilizadas?",
    "¿Qué productos tienen más ventas?",
    "Explica la estructura de costos de almacenamiento",
]


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    from rag.vector_store import load_vector_store
    
    # Load vector store
    store = load_vector_store()
    
    if store is None:
        print("[ERROR] No hay vector store. Ejecuta: python main.py --index")
        exit(1)
    
    # Create chain
    chain = create_rag_chain(store)
    
    # Test with example query
    print("\n" + "=" * 50)
    print("[RAG] RAG Logistics Assistant")
    print("=" * 50)
    
    query = EXAMPLE_QUERIES[0]
    print(f"\n[?] Pregunta: {query}\n")
    
    response = query_rag(chain, query, verbose=True)
    print(f"\n[>] Respuesta:\n{response}")
