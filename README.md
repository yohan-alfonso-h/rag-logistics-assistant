# RAG Logistics Assistant 🚚📦

Un sistema **RAG (Retrieval Augmented Generation)** para consultar información sobre logística y cadena de suministro, usando LangChain, ChromaDB y OpenAI.

## 📋 Características

- 📥 Descarga automática de datasets de logística desde GitHub
- 🔍 Búsqueda semántica con embeddings de OpenAI
- 💬 Interfaz de chat interactiva en español
- 🗄️ Almacenamiento vectorial persistente con ChromaDB

## 🏗️ Arquitectura

```
rag-logistics-assistant/
│
├── data/
│   └── raw/                    # Datasets descargados
│
├── rag/
│   ├── __init__.py
│   ├── data_loader.py          # Descarga y procesamiento de datos
│   ├── vector_store.py         # ChromaDB y embeddings
│   └── chain.py                # Pipeline RAG con LangChain
│
├── notebooks/                  # Jupyter notebooks
├── chroma_db/                  # Base de datos vectorial (auto-generado)
│
├── main.py                     # CLI principal
├── requirements.txt
└── README.md
```

## 🚀 Quickstart

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Configurar API Key

Crea un archivo `.env` en la raíz del proyecto:

```bash
OPENAI_API_KEY=tu-api-key-aqui
```

### 3. Descargar datasets

```bash
python main.py --download
```

### 4. Indexar documentos

```bash
python main.py --index
```

### 5. ¡Listo! Usa el RAG

```bash
# Modo interactivo
python main.py --interactive

# Pregunta específica
python main.py --query "¿Cuáles son los modos de envío más utilizados?"

# Demo con ejemplos
python main.py --demo
```

## 📊 Datasets Incluidos

| Dataset | Descripción | Fuente |
|---------|-------------|--------|
| DataCo Supply Chain | Ventas, envíos y clientes | [GitHub](https://github.com/ashishpatel26/DataCo-SMART-SUPPLY-CHAIN-FOR-BIG-DATA-ANALYSIS) |
| Logistics Problem | Órdenes, puertos y almacenes | [GitHub](https://github.com/jaredbach/LogisticsDataset) |

## 💡 Ejemplos de Preguntas

- ¿Cuáles son los principales modos de envío utilizados?
- ¿Qué carriers manejan las tarifas más bajas?
- Describe los problemas de entrega más comunes
- ¿Cuáles son las rutas de envío más utilizadas?
- ¿Qué productos tienen más ventas?

## 🛠️ Tecnologías

- **LangChain**: Framework para aplicaciones LLM
- **OpenAI**: Embeddings y generación de texto
- **ChromaDB**: Base de datos vectorial
- **Pandas**: Procesamiento de datos

## 📖 Aprendizaje

Este proyecto es ideal para aprender sobre:

1. **Embeddings**: Cómo convertir texto en vectores numéricos
2. **Vector DBs**: Almacenamiento y búsqueda semántica
3. **RAG Pattern**: Combinar retrieval con generación
4. **Prompt Engineering**: Diseño de prompts efectivos
5. **LangChain**: Construcción de pipelines LLM

---

Creado para practicar y aprender IA 🤖
