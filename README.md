# 🔍 Semantic Retrieval System using FAISS

A complete, production-ready semantic retrieval system for RAG applications.

## 📋 Overview

This system:
- **Ingests** raw textual knowledge from `.txt` files
- **Converts** text into dense vector embeddings
- **Stores** embeddings in a FAISS index
- **Retrieves** the most semantically similar text chunks for any query

> **Note**: This is a **RETRIEVER only** — RAG-ready, but without generation.

---

## 🏗️ Architecture

```
Raw Text Files (.txt)
        ↓
Text Normalization (whitespace, newlines)
        ↓
Chunking + Metadata (~225 words, 50 overlap)
        ↓
Embedding Model (all-MiniLM-L6-v2)
        ↓
Vector Embeddings (384 dimensions, normalized)
        ↓
FAISS Index (IndexFlatIP for cosine similarity)
        ↓
Semantic Retrieval (Top-K similar chunks)
```

---

## 📁 Project Structure

```
STIAI/
├── .venv/                   # Virtual environment
├── data/                    # Put your .txt files here
│   ├── document1.txt
│   ├── document2.txt
│   └── ...
├── index/                   # Generated artifacts
│   ├── knowledge.faiss      # FAISS vector index
│   ├── documents.pkl        # Chunk texts
│   └── meta.pkl             # Chunk metadata
├── semantic_retriever.py    # Main Python script
├── requirements.txt         # Dependencies
└── README.md
```

---

## 🚀 Quick Start

### Step 1: Create & Activate Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run the System

```bash
python semantic_retriever.py
```

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `faiss-cpu` | ≥1.7.0 | Vector similarity search |
| `sentence-transformers` | ≥2.2.0 | Text embeddings |
| `numpy` | ≥1.21.0 | Numerical operations |

---

## 🔧 Configuration

Edit the `RetrieverConfig` class to customize:

```python
@dataclass
class RetrieverConfig:
    data_folder: str = "data"           # Input folder
    output_folder: str = "index"        # Output folder
    chunk_size_words: int = 225         # Words per chunk
    chunk_overlap_words: int = 50       # Overlap words
    model_name: str = "all-MiniLM-L6-v2"  # Embedding model
```

---

## 📝 Usage Examples

### Building the Index

```python
from semantic_retriever import build_index_pipeline, RetrieverConfig

config = RetrieverConfig(
    data_folder="data",
    output_folder="index"
)

build_index_pipeline(config)
```

### Retrieving Documents

```python
from semantic_retriever import SemanticRetriever, RetrieverConfig

# Initialize and load
retriever = SemanticRetriever(RetrieverConfig())
retriever.load()

# Query
query = "What is machine learning?"
results = retriever.retrieve(query, top_k=5)

# Display results
for result in results:
    print(f"Score: {result.similarity_score:.4f}")
    print(f"Source: {result.source_file}")
    print(f"Preview: {result.text_preview}")
    print()
```

---

## 📊 Output Format

Each retrieval result contains:

| Field | Type | Description |
|-------|------|-------------|
| `similarity_score` | float | Cosine similarity (0-1) |
| `source_file` | str | Original filename |
| `chunk_id` | int | Chunk index within file |
| `text_preview` | str | First 150 characters |
| `full_text` | str | Complete chunk text |

---

## 📺 Expected Output

When you run `python semantic_retriever.py`, you'll see:

```
📂 STEP 1: DATA INGESTION
============================================================
  ✓ Loaded: machine_learning.txt (1,639 characters)
  ✓ Loaded: deep_learning.txt (1,607 characters)
  ...

🔧 STEP 2: TEXT PREPROCESSING
============================================================
  ✓ machine_learning.txt: 1,639 → 1,580 chars

✂️  STEP 3: TEXT CHUNKING
============================================================
  📐 Chunk size: ~225 words
  ✓ machine_learning.txt: 2 chunks

🧠 STEP 4: LOADING EMBEDDING MODEL
============================================================
  ✓ Model loaded successfully!
  📐 Embedding dimension: 384

🔢 GENERATING EMBEDDINGS
============================================================
  ✓ Generated 6 embeddings

🏗️  STEP 5: BUILDING FAISS INDEX
============================================================
  ✓ Index built successfully!

💾 STEP 6: SAVING ARTIFACTS
============================================================
  ✓ Saved FAISS index → index/knowledge.faiss

🔍 SEMANTIC RETRIEVAL
============================================================
  📝 Query: "What are the different types of machine learning?"
  ✓ Found 3 relevant chunks

📋 RETRIEVAL RESULTS
============================================================
🏆 Result #1
📄 Source: machine_learning.txt | Chunk #0
📊 Similarity: 0.7823
```

---

## ⚠️ Error Handling

| Error | Behavior |
|-------|----------|
| Empty data folder | Raises `ValueError` with clear message |
| Missing index files | Raises `FileNotFoundError` with instructions |
| Invalid FAISS scores | Skips with warning, continues retrieval |
| top_k > documents | Auto-adjusts to available documents |

---

## 🔄 Adding Your Own Data

1. **Delete** sample files in `data/`
2. **Add** your own `.txt` files to `data/`
3. **Run** `python semantic_retriever.py` again

---

## 🔄 Extending to RAG

Integrate with any LLM:

```python
def rag_answer(query, retriever, llm):
    # 1. Retrieve
    results = retriever.retrieve(query, top_k=5)
    
    # 2. Build context
    context = "\n\n".join([r.full_text for r in results])
    
    # 3. Generate
    prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    answer = llm.generate(prompt)
    
    return answer, results
```

---

## 📈 Performance Tips

1. **Larger datasets**: Use `IndexIVFFlat` for faster approximate search
2. **GPU acceleration**: Install `faiss-gpu` instead of `faiss-cpu`
3. **Better embeddings**: Try larger models like `all-mpnet-base-v2`
4. **Chunk tuning**: Adjust chunk size based on your document structure

---

## 📄 License

MIT License - Feel free to use and modify for your projects.

---

**Built with ❤️ for Information Retrieval and RAG systems**
