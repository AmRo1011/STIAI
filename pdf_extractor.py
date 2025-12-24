"""
================================================================================
PDF EXTRACTOR MODULE
================================================================================
Extracts text content from PDF files for the Semantic Retrieval System.

Supports:
- Single and multi-page PDFs
- Text extraction with layout preservation
- Automatic encoding handling
- Error handling for corrupted PDFs
================================================================================
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

# PDF extraction library
try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False


class PDFExtractor:
    """
    Extracts text from PDF files using multiple backends.
    
    Priority:
    1. pdfplumber (better for complex layouts)
    2. PyPDF2 (fallback, widely compatible)
    """
    
    def __init__(self, prefer_pdfplumber: bool = True):
        """
        Initialize the PDF extractor.
        
        Args:
            prefer_pdfplumber: Whether to prefer pdfplumber over PyPDF2
        """
        self.prefer_pdfplumber = prefer_pdfplumber
        self._check_dependencies()
    
    def _check_dependencies(self) -> None:
        """Check if required libraries are available."""
        if not PYPDF2_AVAILABLE and not PDFPLUMBER_AVAILABLE:
            raise ImportError(
                "❌ No PDF library found!\n"
                "Please install one of the following:\n"
                "  pip install pdfplumber  (recommended)\n"
                "  pip install PyPDF2"
            )
        
        if self.prefer_pdfplumber and not PDFPLUMBER_AVAILABLE:
            print("⚠️ pdfplumber not available, falling back to PyPDF2")
            self.prefer_pdfplumber = False
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract all text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as a single string
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            ValueError: If PDF cannot be read
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"❌ PDF file not found: {pdf_path}")
        
        if not pdf_path.suffix.lower() == '.pdf':
            raise ValueError(f"❌ Not a PDF file: {pdf_path}")
        
        # Try preferred method first
        if self.prefer_pdfplumber and PDFPLUMBER_AVAILABLE:
            try:
                return self._extract_with_pdfplumber(pdf_path)
            except Exception as e:
                print(f"⚠️ pdfplumber failed, trying PyPDF2: {e}")
                if PYPDF2_AVAILABLE:
                    return self._extract_with_pypdf2(pdf_path)
                raise
        else:
            return self._extract_with_pypdf2(pdf_path)
    
    def _extract_with_pdfplumber(self, pdf_path: Path) -> str:
        """Extract text using pdfplumber."""
        text_parts = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        
        return "\n\n".join(text_parts)
    
    def _extract_with_pypdf2(self, pdf_path: Path) -> str:
        """Extract text using PyPDF2."""
        text_parts = []
        
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        
        return "\n\n".join(text_parts)
    
    def extract_with_metadata(
        self,
        pdf_path: str
    ) -> Tuple[str, Dict[str, str]]:
        """
        Extract text and metadata from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Tuple of (extracted_text, metadata_dict)
        """
        pdf_path = Path(pdf_path)
        text = self.extract_text_from_pdf(pdf_path)
        
        metadata = {
            'filename': pdf_path.name,
            'file_size': pdf_path.stat().st_size,
            'num_characters': len(text),
            'num_words': len(text.split())
        }
        
        # Try to get PDF metadata
        if PYPDF2_AVAILABLE:
            try:
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    metadata['num_pages'] = len(reader.pages)
                    
                    if reader.metadata:
                        if reader.metadata.title:
                            metadata['title'] = reader.metadata.title
                        if reader.metadata.author:
                            metadata['author'] = reader.metadata.author
            except:
                pass
        
        return text, metadata


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Convenience function to extract text from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text
    """
    extractor = PDFExtractor()
    return extractor.extract_text_from_pdf(pdf_path)


def extract_text_from_pdf_bytes(pdf_bytes: bytes, filename: str = "document.pdf") -> str:
    """
    Extract text from PDF bytes (useful for file uploads).
    
    Args:
        pdf_bytes: PDF file content as bytes
        filename: Optional filename for error messages
        
    Returns:
        Extracted text
    """
    import tempfile
    
    # Write bytes to temp file
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    
    try:
        extractor = PDFExtractor()
        text = extractor.extract_text_from_pdf(tmp_path)
        return text
    finally:
        # Clean up temp file
        os.unlink(tmp_path)


def load_pdf_files(folder_path: str) -> Dict[str, str]:
    """
    Load all PDF files from a folder and extract their text.
    
    Args:
        folder_path: Path to folder containing PDF files
        
    Returns:
        Dictionary mapping filename to extracted text
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        raise FileNotFoundError(f"❌ Folder not found: {folder_path}")
    
    pdf_files = list(folder.glob("*.pdf")) + list(folder.glob("*.PDF"))
    
    if not pdf_files:
        return {}
    
    extractor = PDFExtractor()
    documents = {}
    
    print(f"\n📄 Extracting text from {len(pdf_files)} PDF files...")
    
    for pdf_path in pdf_files:
        try:
            text = extractor.extract_text_from_pdf(pdf_path)
            if text.strip():
                documents[pdf_path.name] = text
                print(f"  ✓ {pdf_path.name}: {len(text):,} characters extracted")
            else:
                print(f"  ⚠️ {pdf_path.name}: No text extracted (might be scanned/image PDF)")
        except Exception as e:
            print(f"  ❌ {pdf_path.name}: Error - {e}")
    
    return documents


def clean_pdf_text(text: str) -> str:
    """
    Clean extracted PDF text.
    
    Operations:
    - Fix common PDF extraction issues
    - Normalize whitespace
    - Remove page numbers patterns
    - Fix hyphenation at line breaks
    
    Args:
        text: Raw extracted text
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Fix hyphenation at line breaks (word- \n continuation)
    text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
    
    # Remove standalone page numbers
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    text = re.sub(r'\n\s*Page\s+\d+\s*(of\s+\d+)?\s*\n', '\n', text, flags=re.IGNORECASE)
    
    # Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Fix common OCR/extraction errors
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    
    # Strip lines
    lines = text.split('\n')
    lines = [line.strip() for line in lines]
    text = '\n'.join(lines)
    
    return text.strip()


# ==============================================================================
# MAIN - Testing
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("📄 PDF EXTRACTOR MODULE")
    print("=" * 60)
    
    print("\n📦 Available libraries:")
    print(f"  • pdfplumber: {'✅ Available' if PDFPLUMBER_AVAILABLE else '❌ Not installed'}")
    print(f"  • PyPDF2: {'✅ Available' if PYPDF2_AVAILABLE else '❌ Not installed'}")
    
    if not PYPDF2_AVAILABLE and not PDFPLUMBER_AVAILABLE:
        print("\n⚠️ Please install a PDF library:")
        print("  pip install pdfplumber")
        print("  # or")
        print("  pip install PyPDF2")
    else:
        print("\n✅ PDF extraction is ready!")
        print("\nUsage:")
        print("  from pdf_extractor import extract_text_from_pdf")
        print("  text = extract_text_from_pdf('document.pdf')")
