"""
Data Loader Module
Downloads and processes logistics datasets from GitHub.
"""

import os
import urllib.request
import pandas as pd
from pathlib import Path
from typing import List
from langchain_core.documents import Document


# Dataset URLs from GitHub
DATASETS = {
    "supply_chain": {
        "url": "https://raw.githubusercontent.com/ashishpatel26/DataCo-SMART-SUPPLY-CHAIN-FOR-BIG-DATA-ANALYSIS/master/DataCoSupplyChainDataset.csv",
        "filename": "supply_chain_dataset.csv",
        "description": "DataCo Supply Chain - Datos de ventas, envios y clientes"
    }
}


def get_data_dir() -> Path:
    """Get the raw data directory path."""
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def download_datasets(force: bool = False) -> dict:
    """
    Download all logistics datasets from GitHub.
    
    Args:
        force: If True, re-download even if files exist
        
    Returns:
        Dictionary with dataset names and their local paths
    """
    data_dir = get_data_dir()
    downloaded = {}
    
    for name, info in DATASETS.items():
        filepath = data_dir / info["filename"]
        
        if filepath.exists() and not force:
            print(f"[OK] {name}: Ya existe en {filepath}")
            downloaded[name] = str(filepath)
            continue
            
        print(f"[DOWNLOAD] Descargando {name}...")
        try:
            urllib.request.urlretrieve(info["url"], filepath)
            print(f"  [OK] Guardado en {filepath}")
            downloaded[name] = str(filepath)
        except Exception as e:
            print(f"  [ERROR] Error descargando {name}: {e}")
            
    return downloaded


def _create_supply_chain_documents(df: pd.DataFrame, max_rows: int = 500) -> List[Document]:
    """Convert supply chain DataFrame to LangChain Documents."""
    documents = []
    
    # Sample rows if too large
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42)
    
    for idx, row in df.iterrows():
        # Create a narrative description of each order
        content = f"""
Orden de Envío #{row.get('Order Id', idx)}
======================================
Cliente: {row.get('Customer Full Name', 'N/A')} ({row.get('Customer Segment', 'N/A')})
Ciudad: {row.get('Customer City', 'N/A')}, {row.get('Customer State', 'N/A')}, {row.get('Customer Country', 'N/A')}

Producto: {row.get('Product Name', 'N/A')}
Categoría: {row.get('Category Name', 'N/A')} > {row.get('Department Name', 'N/A')}
Precio: ${row.get('Product Price', 0):.2f}
Cantidad: {row.get('Order Item Quantity', 1)}

Envío:
- Modo: {row.get('Shipping Mode', 'N/A')}
- Estado: {row.get('Delivery Status', 'N/A')}
- Días programados: {row.get('Days for shipping (scheduled)', 'N/A')}
- Días reales: {row.get('Days for shipment (scheduled)', 'N/A')}

Mercado: {row.get('Market', 'N/A')}
Región: {row.get('Order Region', 'N/A')}
"""
        
        metadata = {
            "source": "supply_chain_dataset",
            "order_id": str(row.get('Order Id', idx)),
            "category": str(row.get('Category Name', '')),
            "shipping_mode": str(row.get('Shipping Mode', '')),
            "market": str(row.get('Market', ''))
        }
        
        documents.append(Document(page_content=content.strip(), metadata=metadata))
    
    return documents


def _create_orders_documents(df: pd.DataFrame) -> List[Document]:
    """Convert orders DataFrame to LangChain Documents."""
    documents = []
    
    for idx, row in df.iterrows():
        content = f"""
Orden de Logística
==================
ID de Orden: {row.get('Order ID', idx)}
Origen: {row.get('Origin Port', 'N/A')}
Destino: Puerto de destino
Planta: {row.get('Plant Code', 'N/A')}

Unidades: {row.get('Unit quantity', 'N/A')}
Peso: {row.get('Weight', 'N/A')} kg

Servicio: {row.get('Service Level', 'N/A')}
Carrier: {row.get('Carrier', 'N/A')}
"""
        
        metadata = {
            "source": "order_list",
            "order_id": str(row.get('Order ID', idx)),
            "origin": str(row.get('Origin Port', '')),
            "plant": str(row.get('Plant Code', ''))
        }
        
        documents.append(Document(page_content=content.strip(), metadata=metadata))
    
    return documents


def _create_freight_documents(df: pd.DataFrame) -> List[Document]:
    """Convert freight rates DataFrame to LangChain Documents."""
    documents = []
    
    for idx, row in df.iterrows():
        content = f"""
Tarifa de Flete
===============
Carrier: {row.get('Carrier', 'N/A')}
Puerto de Origen: {row.get('orig_port_cd', 'N/A')}
Puerto de Destino: {row.get('dest_port_cd', 'N/A')}

Rango de Peso:
- Mínimo: {row.get('minm_wgh_qty', 0)} kg
- Máximo: {row.get('max_wgh_qty', 0)} kg

Tarifa: ${row.get('rate', 0):.2f}
Modo de Transporte: {row.get('mode_dsc', 'N/A')}
Tipo de Servicio: {row.get('svc_cd', 'N/A')}
"""
        
        metadata = {
            "source": "freight_rates",
            "carrier": str(row.get('Carrier', '')),
            "mode": str(row.get('mode_dsc', ''))
        }
        
        documents.append(Document(page_content=content.strip(), metadata=metadata))
    
    return documents


def load_documents(max_supply_chain_rows: int = 500) -> List[Document]:
    """
    Load all datasets and convert them to LangChain Documents.
    
    Args:
        max_supply_chain_rows: Maximum rows to load from the large supply chain dataset
        
    Returns:
        List of Document objects ready for embedding
    """
    data_dir = get_data_dir()
    all_documents = []
    
    # Load Supply Chain Dataset
    supply_chain_path = data_dir / "supply_chain_dataset.csv"
    if supply_chain_path.exists():
        print("[LOAD] Cargando Supply Chain Dataset...")
        try:
            df = pd.read_csv(supply_chain_path, encoding='latin-1')
            docs = _create_supply_chain_documents(df, max_supply_chain_rows)
            all_documents.extend(docs)
            print(f"   [OK] {len(docs)} documentos creados")
        except Exception as e:
            print(f"   [ERROR] Error: {e}")
    
    # Load Orders
    orders_path = data_dir / "order_list.csv"
    if orders_path.exists():
        print("[LOAD] Cargando Order List...")
        try:
            df = pd.read_csv(orders_path)
            docs = _create_orders_documents(df)
            all_documents.extend(docs)
            print(f"   [OK] {len(docs)} documentos creados")
        except Exception as e:
            print(f"   [ERROR] Error: {e}")
    
    # Load Freight Rates
    freight_path = data_dir / "freight_rates.csv"
    if freight_path.exists():
        print("[LOAD] Cargando Freight Rates...")
        try:
            df = pd.read_csv(freight_path)
            docs = _create_freight_documents(df)
            all_documents.extend(docs)
            print(f"   [OK] {len(docs)} documentos creados")
        except Exception as e:
            print(f"   [ERROR] Error: {e}")
    
    print(f"\n[TOTAL] Total: {len(all_documents)} documentos cargados")
    return all_documents


if __name__ == "__main__":
    # Download datasets
    print("=" * 50)
    print("Descargando datasets de logística...")
    print("=" * 50)
    download_datasets()
    
    # Load and show sample
    print("\n" + "=" * 50)
    print("Cargando documentos...")
    print("=" * 50)
    docs = load_documents(max_supply_chain_rows=100)
    
    if docs:
        print("\n[SAMPLE] Ejemplo de documento:")
        print("-" * 40)
        print(docs[0].page_content[:500])
