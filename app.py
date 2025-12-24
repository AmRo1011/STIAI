"""
================================================================================
SEMANTIC RETRIEVAL SYSTEM - STREAMLIT UI
================================================================================
A beautiful web interface for the FAISS-based semantic retrieval system.
Supports both TXT and PDF files.
================================================================================
"""

import os
import streamlit as st
import numpy as np
from pathlib import Path
import tempfile
import shutil

# Import from our retriever module
from semantic_retriever import (
    RetrieverConfig,
    SemanticRetriever,
    load_text_files,
    preprocess_documents,
    chunk_all_documents,
    EmbeddingModel,
    generate_embeddings,
    build_faiss_index,
    save_artifacts,
    TextChunk
)

# Import PDF extractor
try:
    from pdf_extractor import (
        extract_text_from_pdf_bytes,
        clean_pdf_text,
        PYPDF2_AVAILABLE,
        PDFPLUMBER_AVAILABLE
    )
    PDF_SUPPORT = PYPDF2_AVAILABLE or PDFPLUMBER_AVAILABLE
except ImportError:
    PDF_SUPPORT = False

# ==============================================================================
# PAGE CONFIG
# ==============================================================================
st.set_page_config(
    page_title="🔍 Semantic Retrieval System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# CUSTOM CSS
# ==============================================================================
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        color: #a0aec0;
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* Card styling */
    .result-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .result-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.2);
    }
    
    .score-badge {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .source-badge {
        background: rgba(102, 126, 234, 0.2);
        color: #667eea;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
    }
    
    .pdf-badge {
        background: rgba(220, 38, 38, 0.2);
        color: #fc8181;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(26, 26, 46, 0.9);
    }
    
    /* Upload area */
    .upload-area {
        border: 2px dashed rgba(102, 126, 234, 0.5);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        background: rgba(102, 126, 234, 0.05);
        margin-bottom: 1rem;
    }
    
    /* Stats cards */
    .stat-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .stat-label {
        color: #a0aec0;
        font-size: 0.9rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
    }
    
    /* Text input styling */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        color: white;
        padding: 1rem;
    }
    
    /* Success/Info messages */
    .success-message {
        background: rgba(72, 187, 120, 0.1);
        border: 1px solid rgba(72, 187, 120, 0.3);
        border-radius: 12px;
        padding: 1rem;
        color: #48bb78;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
    }
    
    /* File type indicator */
    .file-type-txt {
        color: #48bb78;
    }
    
    .file-type-pdf {
        color: #fc8181;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'is_indexed' not in st.session_state:
    st.session_state.is_indexed = False
if 'num_documents' not in st.session_state:
    st.session_state.num_documents = 0
if 'num_chunks' not in st.session_state:
    st.session_state.num_chunks = 0
if 'uploaded_files_list' not in st.session_state:
    st.session_state.uploaded_files_list = []

# ==============================================================================
# HEADER
# ==============================================================================
st.markdown('<h1 class="main-header">🔍 Semantic Retrieval System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload your documents (TXT or PDF) and ask questions using AI-powered semantic search</p>', unsafe_allow_html=True)

# ==============================================================================
# SIDEBAR - FILE UPLOAD
# ==============================================================================
with st.sidebar:
    st.markdown("## 📁 Document Upload")
    
    # Show supported formats
    if PDF_SUPPORT:
        st.success("✅ PDF & TXT supported")
        supported_types = ['txt', 'pdf']
        file_help = "Upload text (.txt) or PDF (.pdf) files"
    else:
        st.warning("⚠️ PDF support not available. Install pdfplumber or PyPDF2")
        supported_types = ['txt']
        file_help = "Upload text (.txt) files"
    
    st.markdown("---")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload your documents",
        type=supported_types,
        accept_multiple_files=True,
        help=file_help
    )
    
    if uploaded_files:
        st.markdown(f"**📄 {len(uploaded_files)} file(s) selected:**")
        for file in uploaded_files:
            if file.name.lower().endswith('.pdf'):
                st.markdown(f"- 📕 `{file.name}`")
            else:
                st.markdown(f"- 📄 `{file.name}`")
    
    st.markdown("---")
    
    # Index configuration
    st.markdown("## ⚙️ Configuration")
    
    chunk_size = st.slider(
        "Chunk Size (words)",
        min_value=100,
        max_value=500,
        value=225,
        step=25,
        help="Number of words per chunk"
    )
    
    chunk_overlap = st.slider(
        "Chunk Overlap (words)",
        min_value=20,
        max_value=100,
        value=50,
        step=10,
        help="Number of overlapping words between chunks"
    )
    
    top_k = st.slider(
        "Results to retrieve (Top-K)",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of most similar chunks to retrieve"
    )
    
    st.markdown("---")
    
    # Build Index Button
    if st.button("🚀 Build Index", use_container_width=True, type="primary"):
        if not uploaded_files:
            st.error("❌ Please upload at least one file!")
        else:
            with st.spinner("🔄 Building index..."):
                try:
                    # Create temp directory for uploaded files
                    temp_dir = tempfile.mkdtemp()
                    data_dir = os.path.join(temp_dir, "data")
                    index_dir = os.path.join(temp_dir, "index")
                    os.makedirs(data_dir, exist_ok=True)
                    os.makedirs(index_dir, exist_ok=True)
                    
                    # Process uploaded files
                    documents = {}
                    pdf_count = 0
                    txt_count = 0
                    
                    for file in uploaded_files:
                        file_name = file.name
                        file_bytes = file.getvalue()
                        
                        if file_name.lower().endswith('.pdf'):
                            # Extract text from PDF
                            if PDF_SUPPORT:
                                try:
                                    text = extract_text_from_pdf_bytes(file_bytes, file_name)
                                    text = clean_pdf_text(text)
                                    if text.strip():
                                        # Save as .txt for processing
                                        txt_name = file_name.rsplit('.', 1)[0] + '.txt'
                                        documents[txt_name] = text
                                        pdf_count += 1
                                        st.toast(f"✅ Extracted: {file_name}")
                                    else:
                                        st.warning(f"⚠️ No text in: {file_name}")
                                except Exception as e:
                                    st.error(f"❌ PDF error ({file_name}): {e}")
                            else:
                                st.error(f"❌ PDF not supported: {file_name}")
                        else:
                            # Regular text file
                            try:
                                text = file_bytes.decode('utf-8')
                                documents[file_name] = text
                                txt_count += 1
                            except UnicodeDecodeError:
                                try:
                                    text = file_bytes.decode('latin-1')
                                    documents[file_name] = text
                                    txt_count += 1
                                except:
                                    st.error(f"❌ Encoding error: {file_name}")
                    
                    if not documents:
                        st.error("❌ No valid documents to process!")
                    else:
                        # Save documents to temp directory
                        for name, content in documents.items():
                            file_path = os.path.join(data_dir, name)
                            with open(file_path, 'w', encoding='utf-8') as f:
                                f.write(content)
                        
                        # Build index
                        config = RetrieverConfig(
                            data_folder=data_dir,
                            output_folder=index_dir,
                            chunk_size_words=chunk_size,
                            chunk_overlap_words=chunk_overlap
                        )
                        
                        # Preprocess and chunk
                        preprocessed = preprocess_documents(documents)
                        chunks = chunk_all_documents(
                            preprocessed,
                            chunk_size=chunk_size,
                            overlap=chunk_overlap
                        )
                        
                        # Generate embeddings
                        model = EmbeddingModel(config.model_name)
                        embeddings = generate_embeddings(chunks, model)
                        
                        # Build and save index
                        index = build_faiss_index(embeddings)
                        save_artifacts(index, chunks, config)
                        
                        # Load retriever
                        retriever = SemanticRetriever(config)
                        retriever.load()
                        
                        # Update session state
                        st.session_state.retriever = retriever
                        st.session_state.is_indexed = True
                        st.session_state.num_documents = len(documents)
                        st.session_state.num_chunks = len(chunks)
                        st.session_state.uploaded_files_list = [
                            (f.name, 'pdf' if f.name.lower().endswith('.pdf') else 'txt')
                            for f in uploaded_files
                        ]
                        
                        st.success(f"✅ Index built! ({txt_count} TXT, {pdf_count} PDF)")
                    
                except Exception as e:
                    st.error(f"❌ Error building index: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    # Status
    st.markdown("---")
    st.markdown("## 📊 Status")
    
    if st.session_state.is_indexed:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{st.session_state.num_documents}</div>
            <div class="stat-label">Documents</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="stat-card" style="margin-top: 0.5rem;">
            <div class="stat-number">{st.session_state.num_chunks}</div>
            <div class="stat-label">Chunks</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**📄 Indexed files:**")
        for fname, ftype in st.session_state.uploaded_files_list:
            if ftype == 'pdf':
                st.markdown(f"- ✅ 📕 `{fname}`")
            else:
                st.markdown(f"- ✅ 📄 `{fname}`")
    else:
        st.info("📥 Upload files and build index to start")

# ==============================================================================
# MAIN CONTENT - QUERY INTERFACE
# ==============================================================================
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("## 💬 Ask a Question")
    
    query = st.text_input(
        "Enter your question:",
        placeholder="What is machine learning?",
        label_visibility="collapsed"
    )

with col2:
    st.markdown("## &nbsp;")  # Spacer
    search_button = st.button("🔍 Search", use_container_width=True, type="primary")

# ==============================================================================
# SEARCH RESULTS
# ==============================================================================
if search_button:
    if not st.session_state.is_indexed:
        st.warning("⚠️ Please upload documents and build the index first!")
    elif not query.strip():
        st.warning("⚠️ Please enter a question!")
    else:
        with st.spinner("🔍 Searching..."):
            try:
                results = st.session_state.retriever.retrieve(query, top_k=top_k)
                
                if results:
                    st.markdown("---")
                    st.markdown(f"## 📋 Results ({len(results)} matches)")
                    
                    for i, result in enumerate(results, 1):
                        # Determine if source was PDF
                        is_pdf = any(
                            fname.rsplit('.', 1)[0] + '.txt' == result.source_file
                            for fname, ftype in st.session_state.uploaded_files_list
                            if ftype == 'pdf'
                        )
                        
                        with st.container():
                            source_badge = f'<span class="pdf-badge">📕 {result.source_file}</span>' if is_pdf else f'<span class="source-badge">📄 {result.source_file}</span>'
                            
                            st.markdown(f"""
                            <div class="result-card">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                                    <span style="font-size: 1.2rem; font-weight: 600; color: white;">🏆 Result #{i}</span>
                                    <div>
                                        <span class="score-badge">Score: {result.similarity_score:.4f}</span>
                                        {source_badge}
                                    </div>
                                </div>
                                <div style="color: #a0aec0; font-size: 0.9rem; margin-bottom: 0.5rem;">
                                    Chunk #{result.chunk_id}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            with st.expander("📖 View Full Text", expanded=(i == 1)):
                                st.markdown(result.full_text)
                            
                            st.markdown("")  # Spacer
                else:
                    st.info("No results found. Try a different query.")
                    
            except Exception as e:
                st.error(f"❌ Error during search: {str(e)}")

# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #a0aec0; padding: 2rem;">
    <p>🔍 <strong>Semantic Retrieval System</strong> | Built with FAISS + Sentence Transformers</p>
    <p style="font-size: 0.8rem;">RAG-ready retrieval system for semantic search | Supports TXT & PDF files</p>
</div>
""", unsafe_allow_html=True)
