"""
================================================================================
SEMANTIC RETRIEVAL SYSTEM USING FAISS AND SENTENCE TRANSFORMERS
================================================================================

A complete retrieval pipeline for RAG-ready applications.
This system ingests text documents, converts them to embeddings, stores them
in a FAISS index, and retrieves the most semantically similar chunks for queries.

Author: Senior ML Engineer
Compatible with: Google Colab, Python 3.8+
================================================================================
"""

# ==============================================================================
# INSTALLATION (Run this first in Google Colab)
# ==============================================================================
# !pip install faiss-cpu sentence-transformers numpy

import os
import re
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field

# These imports will be available after pip install
try:
    import faiss
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("=" * 60)
    print("INSTALLATION REQUIRED")
    print("=" * 60)
    print("Run the following command to install dependencies:")
    print("!pip install faiss-cpu sentence-transformers numpy")
    print("=" * 60)
    raise


# ==============================================================================
# CONFIGURATION
# ==============================================================================
@dataclass
class RetrieverConfig:
    """Configuration for the semantic retriever."""
    
    # Paths
    data_folder: str = "data"
    output_folder: str = "index"
    
    # Chunking parameters
    chunk_size_words: int = 225  # Target ~200-250 words
    chunk_overlap_words: int = 50  # Overlap ~40-60 words
    
    # Embedding model
    model_name: str = "all-MiniLM-L6-v2"
    
    # Artifact filenames
    faiss_index_file: str = "knowledge.faiss"
    documents_file: str = "documents.pkl"
    metadata_file: str = "meta.pkl"
    
    # Retrieval defaults
    default_top_k: int = 5
    preview_length: int = 150  # Characters for text preview
    
    def __post_init__(self):
        """Create output folder if it doesn't exist."""
        os.makedirs(self.output_folder, exist_ok=True)


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================
@dataclass
class TextChunk:
    """Represents a chunk of text with its metadata."""
    text: str
    source_file: str
    chunk_index: int
    preview: str = field(default="")
    
    def __post_init__(self):
        """Generate preview if not provided."""
        if not self.preview:
            self.preview = self.text[:150].replace('\n', ' ').strip() + "..."


@dataclass
class RetrievalResult:
    """Represents a single retrieval result."""
    similarity_score: float
    source_file: str
    chunk_id: int
    text_preview: str
    full_text: str
    
    def __repr__(self):
        return (
            f"\n{'─' * 60}\n"
            f"📄 Source: {self.source_file} | Chunk #{self.chunk_id}\n"
            f"📊 Similarity: {self.similarity_score:.4f}\n"
            f"📝 Preview: {self.text_preview}\n"
            f"{'─' * 60}"
        )


# ==============================================================================
# STEP 1: DATA INGESTION
# ==============================================================================
def load_text_files(data_folder: str) -> Dict[str, str]:
    """
    Load all .txt files from the specified folder.
    
    Args:
        data_folder: Path to the folder containing .txt files
        
    Returns:
        Dictionary mapping filename to file content
        
    Raises:
        FileNotFoundError: If data folder doesn't exist
        ValueError: If no .txt files are found
    """
    print("\n" + "=" * 60)
    print("📂 STEP 1: DATA INGESTION")
    print("=" * 60)
    
    data_path = Path(data_folder)
    
    # Check if folder exists
    if not data_path.exists():
        raise FileNotFoundError(
            f"❌ Data folder '{data_folder}' does not exist.\n"
            f"Please create it and add .txt files to it."
        )
    
    # Find all .txt files
    txt_files = list(data_path.glob("*.txt"))
    
    if not txt_files:
        raise ValueError(
            f"❌ No .txt files found in '{data_folder}'.\n"
            f"Please add text files to the data folder."
        )
    
    # Load each file
    documents = {}
    total_chars = 0
    
    for file_path in txt_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                documents[file_path.name] = content
                total_chars += len(content)
                print(f"  ✓ Loaded: {file_path.name} ({len(content):,} characters)")
        except Exception as e:
            print(f"  ⚠️ Error loading {file_path.name}: {e}")
    
    if not documents:
        raise ValueError("❌ Failed to load any documents. Check file permissions.")
    
    print(f"\n📊 Summary: {len(documents)} files loaded, {total_chars:,} total characters")
    
    return documents


# ==============================================================================
# STEP 2: TEXT PREPROCESSING
# ==============================================================================
def preprocess_text(text: str) -> str:
    """
    Normalize text while preserving semantic meaning.
    
    Operations:
    - Normalize whitespace (multiple spaces → single space)
    - Remove excessive newlines (3+ newlines → 2 newlines)
    - Strip leading/trailing whitespace
    - Preserve punctuation
    - Do NOT remove stopwords
    - Do NOT apply stemming/lemmatization
    
    Args:
        text: Raw text to preprocess
        
    Returns:
        Cleaned text
    """
    if not text or not text.strip():
        return ""
    
    # Replace multiple spaces with single space
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Replace 3+ newlines with exactly 2 newlines (paragraph break)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Replace single newlines within paragraphs with space (except paragraph breaks)
    # This preserves paragraph structure while normalizing line breaks
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    
    # Clean up any resulting multiple spaces
    text = re.sub(r'[ ]+', ' ', text)
    
    # Strip whitespace from each line
    lines = text.split('\n')
    lines = [line.strip() for line in lines]
    text = '\n'.join(lines)
    
    # Final strip
    text = text.strip()
    
    return text


def preprocess_documents(documents: Dict[str, str]) -> Dict[str, str]:
    """
    Preprocess all documents.
    
    Args:
        documents: Dictionary of filename → raw text
        
    Returns:
        Dictionary of filename → preprocessed text
    """
    print("\n" + "=" * 60)
    print("🔧 STEP 2: TEXT PREPROCESSING")
    print("=" * 60)
    
    preprocessed = {}
    
    for filename, text in documents.items():
        original_len = len(text)
        cleaned = preprocess_text(text)
        preprocessed[filename] = cleaned
        
        reduction = (1 - len(cleaned) / original_len) * 100 if original_len > 0 else 0
        print(f"  ✓ {filename}: {original_len:,} → {len(cleaned):,} chars ({reduction:.1f}% reduction)")
    
    print("\n📊 All documents preprocessed successfully")
    
    return preprocessed


# ==============================================================================
# STEP 3: TEXT CHUNKING
# ==============================================================================
def chunk_text(
    text: str,
    source_file: str,
    chunk_size: int = 225,
    overlap: int = 50
) -> List[TextChunk]:
    """
    Split text into overlapping chunks based on word count.
    
    Args:
        text: Text to chunk
        source_file: Source filename for metadata
        chunk_size: Target number of words per chunk (200-250)
        overlap: Number of overlapping words between chunks (40-60)
        
    Returns:
        List of TextChunk objects
    """
    if not text.strip():
        return []
    
    # Split into words while preserving structure
    words = text.split()
    
    if len(words) == 0:
        return []
    
    chunks = []
    start = 0
    chunk_index = 0
    
    while start < len(words):
        # Calculate end position
        end = min(start + chunk_size, len(words))
        
        # Extract chunk words
        chunk_words = words[start:end]
        chunk_text = ' '.join(chunk_words)
        
        # Create chunk object
        chunk = TextChunk(
            text=chunk_text,
            source_file=source_file,
            chunk_index=chunk_index
        )
        chunks.append(chunk)
        
        # Move start position (with overlap)
        start = end - overlap if end < len(words) else len(words)
        chunk_index += 1
        
        # Safety check to prevent infinite loop
        if start >= len(words) or end == len(words):
            break
    
    return chunks


def chunk_all_documents(
    documents: Dict[str, str],
    chunk_size: int = 225,
    overlap: int = 50
) -> List[TextChunk]:
    """
    Chunk all documents in the collection.
    
    Args:
        documents: Dictionary of filename → preprocessed text
        chunk_size: Target words per chunk
        overlap: Overlap words between chunks
        
    Returns:
        List of all TextChunk objects
    """
    print("\n" + "=" * 60)
    print("✂️  STEP 3: TEXT CHUNKING")
    print("=" * 60)
    print(f"  📐 Chunk size: ~{chunk_size} words")
    print(f"  🔗 Overlap: ~{overlap} words")
    print()
    
    all_chunks = []
    
    for filename, text in documents.items():
        chunks = chunk_text(text, filename, chunk_size, overlap)
        all_chunks.extend(chunks)
        
        if chunks:
            avg_words = sum(len(c.text.split()) for c in chunks) / len(chunks)
            print(f"  ✓ {filename}: {len(chunks)} chunks (avg {avg_words:.0f} words/chunk)")
        else:
            print(f"  ⚠️ {filename}: No chunks generated (empty or too short)")
    
    if not all_chunks:
        raise ValueError("❌ No chunks were generated. Check if documents contain text.")
    
    print(f"\n📊 Total chunks generated: {len(all_chunks)}")
    
    return all_chunks


# ==============================================================================
# STEP 4: EMBEDDING GENERATION
# ==============================================================================
class EmbeddingModel:
    """Wrapper for the sentence transformer embedding model."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the sentence-transformers model
        """
        print("\n" + "=" * 60)
        print("🧠 STEP 4: LOADING EMBEDDING MODEL")
        print("=" * 60)
        print(f"  📦 Model: {model_name}")
        print("  ⏳ Loading model (this may take a moment)...")
        
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        print(f"  ✓ Model loaded successfully!")
        print(f"  📐 Embedding dimension: {self.embedding_dim}")
    
    def encode(
        self,
        texts: List[str],
        normalize: bool = True,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            normalize: Whether to L2-normalize embeddings (for cosine similarity)
            show_progress: Whether to show progress bar
            
        Returns:
            NumPy array of shape (n_texts, embedding_dim), dtype float32
        """
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress
        )
        
        # Ensure float32 dtype
        embeddings = embeddings.astype(np.float32)
        
        return embeddings
    
    def encode_single(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text string to embed
            normalize: Whether to L2-normalize embedding
            
        Returns:
            NumPy array of shape (embedding_dim,), dtype float32
        """
        return self.encode([text], normalize=normalize, show_progress=False)[0]


def generate_embeddings(
    chunks: List[TextChunk],
    model: EmbeddingModel
) -> np.ndarray:
    """
    Generate embeddings for all text chunks.
    
    Args:
        chunks: List of TextChunk objects
        model: EmbeddingModel instance
        
    Returns:
        NumPy array of embeddings, shape (n_chunks, embedding_dim)
    """
    print("\n" + "=" * 60)
    print("🔢 GENERATING EMBEDDINGS")
    print("=" * 60)
    print(f"  📝 Processing {len(chunks)} chunks...")
    
    texts = [chunk.text for chunk in chunks]
    embeddings = model.encode(texts, normalize=True, show_progress=True)
    
    print(f"\n  ✓ Generated {embeddings.shape[0]} embeddings")
    print(f"  📐 Shape: {embeddings.shape}")
    print(f"  💾 Memory: {embeddings.nbytes / 1024 / 1024:.2f} MB")
    
    return embeddings


# ==============================================================================
# STEP 5: FAISS INDEX BUILDING
# ==============================================================================
def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a FAISS index for similarity search.
    
    Uses IndexFlatIP (Inner Product) with normalized vectors,
    which is equivalent to cosine similarity.
    
    Args:
        embeddings: NumPy array of normalized embeddings
        
    Returns:
        FAISS index ready for search
    """
    print("\n" + "=" * 60)
    print("🏗️  STEP 5: BUILDING FAISS INDEX")
    print("=" * 60)
    
    # Get dimension from embeddings
    dimension = embeddings.shape[1]
    n_vectors = embeddings.shape[0]
    
    print(f"  📐 Vector dimension: {dimension}")
    print(f"  📊 Number of vectors: {n_vectors}")
    
    # Create index (Inner Product for cosine similarity with normalized vectors)
    index = faiss.IndexFlatIP(dimension)
    
    # Add vectors to index
    print("  ⏳ Adding vectors to index...")
    index.add(embeddings)
    
    print(f"  ✓ Index built successfully!")
    print(f"  📊 Index contains {index.ntotal} vectors")
    
    return index


# ==============================================================================
# STEP 6: SAVING & LOADING ARTIFACTS
# ==============================================================================
def save_artifacts(
    index: faiss.IndexFlatIP,
    chunks: List[TextChunk],
    config: RetrieverConfig
) -> None:
    """
    Save all retrieval artifacts to disk.
    
    Saves:
    - FAISS index → knowledge.faiss
    - Chunk texts → documents.pkl
    - Metadata → meta.pkl
    
    Args:
        index: FAISS index
        chunks: List of TextChunk objects
        config: Retriever configuration
    """
    print("\n" + "=" * 60)
    print("💾 STEP 6: SAVING ARTIFACTS")
    print("=" * 60)
    
    output_path = Path(config.output_folder)
    
    # Save FAISS index
    faiss_path = output_path / config.faiss_index_file
    faiss.write_index(index, str(faiss_path))
    print(f"  ✓ Saved FAISS index → {faiss_path}")
    
    # Save document texts
    documents = [chunk.text for chunk in chunks]
    docs_path = output_path / config.documents_file
    with open(docs_path, 'wb') as f:
        pickle.dump(documents, f)
    print(f"  ✓ Saved documents → {docs_path}")
    
    # Save metadata
    metadata = [
        {
            'source_file': chunk.source_file,
            'chunk_index': chunk.chunk_index,
            'preview': chunk.preview
        }
        for chunk in chunks
    ]
    meta_path = output_path / config.metadata_file
    with open(meta_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"  ✓ Saved metadata → {meta_path}")
    
    # Calculate total size
    total_size = sum(
        os.path.getsize(output_path / f)
        for f in [config.faiss_index_file, config.documents_file, config.metadata_file]
    )
    print(f"\n📊 Total artifact size: {total_size / 1024:.2f} KB")


def load_artifacts(config: RetrieverConfig) -> Tuple[faiss.IndexFlatIP, List[str], List[Dict]]:
    """
    Load retrieval artifacts from disk.
    
    Args:
        config: Retriever configuration
        
    Returns:
        Tuple of (FAISS index, document texts, metadata)
        
    Raises:
        FileNotFoundError: If any artifact is missing
    """
    output_path = Path(config.output_folder)
    
    # Check all files exist
    required_files = [
        config.faiss_index_file,
        config.documents_file,
        config.metadata_file
    ]
    
    for filename in required_files:
        filepath = output_path / filename
        if not filepath.exists():
            raise FileNotFoundError(
                f"❌ Missing artifact: {filepath}\n"
                f"Please run the indexing pipeline first."
            )
    
    # Load FAISS index
    faiss_path = output_path / config.faiss_index_file
    index = faiss.read_index(str(faiss_path))
    
    # Load documents
    docs_path = output_path / config.documents_file
    with open(docs_path, 'rb') as f:
        documents = pickle.load(f)
    
    # Load metadata
    meta_path = output_path / config.metadata_file
    with open(meta_path, 'rb') as f:
        metadata = pickle.load(f)
    
    return index, documents, metadata


# ==============================================================================
# STEP 7: SEMANTIC RETRIEVAL
# ==============================================================================
class SemanticRetriever:
    """
    Main retrieval interface for semantic search.
    
    This class provides methods to search for semantically similar
    text chunks given a natural language query.
    """
    
    def __init__(self, config: Optional[RetrieverConfig] = None):
        """
        Initialize the retriever.
        
        Args:
            config: Retriever configuration (uses defaults if None)
        """
        self.config = config or RetrieverConfig()
        self.index = None
        self.documents = None
        self.metadata = None
        self.model = None
        self._is_loaded = False
    
    def load(self) -> 'SemanticRetriever':
        """
        Load the retrieval artifacts and embedding model.
        
        Returns:
            Self for method chaining
            
        Raises:
            FileNotFoundError: If artifacts are missing
        """
        print("\n" + "=" * 60)
        print("📂 LOADING RETRIEVAL SYSTEM")
        print("=" * 60)
        
        # Load artifacts
        print("  ⏳ Loading artifacts...")
        self.index, self.documents, self.metadata = load_artifacts(self.config)
        print(f"  ✓ Loaded {len(self.documents)} documents from index")
        
        # Load embedding model
        print("  ⏳ Loading embedding model...")
        self.model = EmbeddingModel(self.config.model_name)
        
        self._is_loaded = True
        print("\n✅ Retrieval system ready!")
        
        return self
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """
        Retrieve the most similar chunks for a query.
        
        Args:
            query: Natural language query
            top_k: Number of results to return
            
        Returns:
            List of RetrievalResult objects, sorted by similarity (descending)
            
        Raises:
            RuntimeError: If retriever hasn't been loaded
            ValueError: If top_k exceeds available documents
        """
        if not self._is_loaded:
            raise RuntimeError(
                "❌ Retriever not loaded. Call .load() first."
            )
        
        if not query or not query.strip():
            raise ValueError("❌ Query cannot be empty.")
        
        # Validate top_k
        n_docs = len(self.documents)
        if top_k > n_docs:
            print(f"  ⚠️ Requested top_k={top_k} exceeds {n_docs} documents. Using top_k={n_docs}")
            top_k = n_docs
        
        if top_k < 1:
            raise ValueError("❌ top_k must be at least 1.")
        
        print("\n" + "=" * 60)
        print("🔍 SEMANTIC RETRIEVAL")
        print("=" * 60)
        print(f"  📝 Query: \"{query}\"")
        print(f"  🎯 Top-K: {top_k}")
        
        # Encode query
        print("  ⏳ Encoding query...")
        query_embedding = self.model.encode_single(query, normalize=True)
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        
        # Search
        print("  ⏳ Searching index...")
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Build results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            # Handle invalid indices (can happen with empty index)
            if idx < 0 or idx >= len(self.documents):
                continue
            
            # Handle invalid scores
            if np.isinf(score) or np.isnan(score):
                print(f"  ⚠️ Skipping result {i+1}: invalid score ({score})")
                continue
            
            meta = self.metadata[idx]
            result = RetrievalResult(
                similarity_score=float(score),
                source_file=meta['source_file'],
                chunk_id=meta['chunk_index'],
                text_preview=meta['preview'],
                full_text=self.documents[idx]
            )
            results.append(result)
        
        print(f"\n  ✓ Found {len(results)} relevant chunks")
        
        return results
    
    def display_results(self, results: List[RetrievalResult]) -> None:
        """
        Display retrieval results in a formatted way.
        
        Args:
            results: List of RetrievalResult objects
        """
        if not results:
            print("\n⚠️ No results found.")
            return
        
        print("\n" + "=" * 60)
        print("📋 RETRIEVAL RESULTS")
        print("=" * 60)
        
        for i, result in enumerate(results, 1):
            print(f"\n🏆 Result #{i}")
            print(result)


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def build_index_pipeline(config: Optional[RetrieverConfig] = None) -> None:
    """
    Execute the full indexing pipeline.
    
    This function:
    1. Loads text files from the data folder
    2. Preprocesses the text
    3. Chunks the text with overlap
    4. Generates embeddings
    5. Builds FAISS index
    6. Saves all artifacts
    
    Args:
        config: Retriever configuration (uses defaults if None)
    """
    config = config or RetrieverConfig()
    
    print("\n" + "=" * 70)
    print("🚀 SEMANTIC RETRIEVAL SYSTEM - INDEXING PIPELINE")
    print("=" * 70)
    print(f"  📁 Data folder: {config.data_folder}")
    print(f"  📂 Output folder: {config.output_folder}")
    print(f"  🧠 Model: {config.model_name}")
    
    try:
        # Step 1: Load documents
        documents = load_text_files(config.data_folder)
        
        # Step 2: Preprocess
        preprocessed = preprocess_documents(documents)
        
        # Step 3: Chunk
        chunks = chunk_all_documents(
            preprocessed,
            chunk_size=config.chunk_size_words,
            overlap=config.chunk_overlap_words
        )
        
        # Step 4: Load embedding model
        model = EmbeddingModel(config.model_name)
        
        # Step 5: Generate embeddings
        embeddings = generate_embeddings(chunks, model)
        
        # Step 6: Build index
        index = build_faiss_index(embeddings)
        
        # Step 7: Save artifacts
        save_artifacts(index, chunks, config)
        
        print("\n" + "=" * 70)
        print("✅ INDEXING PIPELINE COMPLETE!")
        print("=" * 70)
        print(f"\n📊 Summary:")
        print(f"  • Documents processed: {len(documents)}")
        print(f"  • Chunks created: {len(chunks)}")
        print(f"  • Vectors indexed: {index.ntotal}")
        print(f"\n📁 Artifacts saved to: {config.output_folder}/")
        print(f"  • {config.faiss_index_file}")
        print(f"  • {config.documents_file}")
        print(f"  • {config.metadata_file}")
        
    except Exception as e:
        print(f"\n❌ PIPELINE FAILED: {e}")
        raise


def demo_retrieval(queries: Optional[List[str]] = None, top_k: int = 3) -> None:
    """
    Demonstrate the retrieval system with example queries.
    
    Args:
        queries: List of query strings (uses defaults if None)
        top_k: Number of results per query
    """
    print("\n" + "=" * 70)
    print("🔍 RETRIEVAL DEMONSTRATION")
    print("=" * 70)
    
    # Default example queries
    if queries is None:
        queries = [
            "What is machine learning?",
            "How does artificial intelligence work?",
            "Tell me about deep learning"
        ]
    
    # Initialize retriever
    config = RetrieverConfig()
    retriever = SemanticRetriever(config)
    retriever.load()
    
    # Run queries
    for query in queries:
        results = retriever.retrieve(query, top_k=top_k)
        retriever.display_results(results)
        print("\n")


# ==============================================================================
# COLAB-READY EXECUTION BLOCK
# ==============================================================================
def create_sample_data():
    """
    Create sample data files for testing.
    This is useful for demonstrating the system without real documents.
    """
    os.makedirs("data", exist_ok=True)
    
    sample_texts = {
        "machine_learning.txt": """
Machine Learning: A Comprehensive Overview

Machine learning is a subset of artificial intelligence that enables systems to learn 
and improve from experience without being explicitly programmed. It focuses on developing 
computer programs that can access data and use it to learn for themselves.

The process of learning begins with observations or data, such as examples, direct 
experience, or instruction, to look for patterns in data and make better decisions in 
the future. The primary aim is to allow computers to learn automatically without human 
intervention or assistance and adjust actions accordingly.

Types of Machine Learning:

1. Supervised Learning: The algorithm learns from labeled training data, and makes 
predictions based on that data. For example, after learning from examples of spam emails, 
the algorithm can identify new spam emails.

2. Unsupervised Learning: The algorithm learns from unlabeled data and tries to find 
hidden patterns or intrinsic structures. Clustering and association are common techniques.

3. Reinforcement Learning: The algorithm learns to perform actions based on rewards 
and punishments. It's widely used in robotics, gaming, and navigation.

Key algorithms include linear regression for predicting continuous values, decision 
trees for classification, neural networks for complex pattern recognition, and support 
vector machines for binary classification.

Machine learning applications are everywhere: from recommendation systems on Netflix 
and Amazon, to fraud detection in banking, to voice recognition in virtual assistants 
like Siri and Alexa.
        """,
        
        "deep_learning.txt": """
Deep Learning: The Foundation of Modern AI

Deep learning is a subset of machine learning that uses artificial neural networks 
with multiple layers to progressively extract higher-level features from raw input. 
For example, in image processing, lower layers may identify edges, while higher layers 
may identify concepts relevant to humans such as digits, letters, or faces.

The term "deep" refers to the number of layers in the neural network. While traditional 
neural networks contain only 2-3 hidden layers, deep networks can have hundreds.

Key Architectures:

Convolutional Neural Networks (CNNs): Primarily used for image recognition and 
computer vision tasks. CNNs use convolutional layers to automatically detect features 
like edges, textures, and shapes.

Recurrent Neural Networks (RNNs): Designed for sequential data like time series or 
natural language. They have connections that form directed cycles, allowing information 
to persist.

Transformers: A revolutionary architecture that uses self-attention mechanisms. 
Transformers have become the foundation for large language models like GPT and BERT, 
enabling breakthrough performance in natural language processing.

Generative Adversarial Networks (GANs): Consist of two networks (generator and 
discriminator) that compete against each other. GANs are capable of generating highly 
realistic images, videos, and audio.

Deep learning has achieved remarkable success in image classification, speech 
recognition, natural language processing, drug discovery, medical diagnosis, and 
autonomous vehicles.
        """,
        
        "natural_language_processing.txt": """
Natural Language Processing: Bridging Humans and Machines

Natural Language Processing (NLP) is a field at the intersection of computer science, 
artificial intelligence, and linguistics. Its goal is to enable computers to understand, 
interpret, and generate human language in a valuable way.

Core NLP Tasks:

Text Classification: Categorizing text into predefined categories. Applications include 
spam detection, sentiment analysis, and topic labeling.

Named Entity Recognition (NER): Identifying and classifying named entities in text 
into categories such as person names, organizations, locations, and dates.

Part-of-Speech Tagging: Assigning parts of speech to each word in a sentence, such 
as noun, verb, adjective, etc.

Machine Translation: Automatically translating text from one language to another. 
Modern systems use neural machine translation with encoder-decoder architectures.

Question Answering: Building systems that can automatically answer questions posed 
in natural language. This includes extractive QA (finding answers in documents) and 
generative QA (generating answers).

Text Summarization: Automatically creating a shorter version of a text while 
preserving key information. Can be extractive or abstractive.

The Transformer Revolution:

The introduction of the Transformer architecture in 2017 revolutionized NLP. 
Pre-trained language models like BERT, GPT, and their successors have achieved 
state-of-the-art performance across virtually all NLP tasks.

These models are first pre-trained on massive amounts of text data to learn language 
patterns, then fine-tuned on specific tasks. This transfer learning approach has 
made it possible to achieve excellent results even with limited task-specific data.

RAG (Retrieval-Augmented Generation) combines retrieval systems with language models 
to ground generations in relevant documents, improving accuracy and reducing 
hallucinations in AI responses.
        """
    }
    
    print("📝 Creating sample data files...")
    for filename, content in sample_texts.items():
        filepath = os.path.join("data", filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content.strip())
        print(f"  ✓ Created: {filepath}")
    
    print("\n✅ Sample data created successfully!")


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║           SEMANTIC RETRIEVAL SYSTEM - FAISS + EMBEDDINGS             ║
║                                                                      ║
║  A RAG-ready retrieval system for semantic search                    ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Check if data folder exists and has files
    if not os.path.exists("data") or not list(Path("data").glob("*.txt")):
        print("⚠️ No data files found. Creating sample data...")
        create_sample_data()
    
    # Build the index
    config = RetrieverConfig(
        data_folder="data",
        output_folder="index",
        chunk_size_words=225,
        chunk_overlap_words=50,
        model_name="all-MiniLM-L6-v2"
    )
    
    build_index_pipeline(config)
    
    # Demonstrate retrieval
    example_queries = [
        "What are the different types of machine learning?",
        "How do transformers work in NLP?",
        "What is the difference between CNN and RNN?"
    ]
    
    demo_retrieval(queries=example_queries, top_k=3)
