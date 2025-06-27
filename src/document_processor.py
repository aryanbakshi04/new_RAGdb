import os
import logging
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .config import Config

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles PDF processing, text extraction, and chunking"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
        )
        # Track processed PDFs
        self.processed_pdfs = set()
    
    def process_pdf(self, pdf_path: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Process a PDF file and split it into chunks for embedding
        Returns a list of document dictionaries with text and metadata
        """
        try:
            logger.info(f"Processing PDF: {pdf_path}")
            
            # Check if PDF exists
            if not os.path.exists(pdf_path):
                logger.error(f"PDF file not found: {pdf_path}")
                return []
                
            # Check if already processed
            pdf_name = os.path.basename(pdf_path)
            if pdf_name in self.processed_pdfs:
                logger.info(f"PDF already processed: {pdf_name}")
                return []
                
            # Load PDF
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            
            if not pages:
                logger.warning(f"No content extracted from {pdf_path}")
                return []
                
            # Get text from all pages
            full_text = "\n".join(page.page_content for page in pages)
            
            if not full_text.strip():
                logger.warning(f"Extracted empty text from {pdf_path}")
                return []
                
            # Split into chunks
            text_chunks = self.text_splitter.split_text(full_text)
            
            # Create document dictionaries
            documents = []
            
            # Try to load metadata from JSON file if it exists
            if not metadata:
                metadata = {}
                json_path = os.path.splitext(pdf_path)[0] + ".json"
                if os.path.exists(json_path):
                    try:
                        with open(json_path, 'r') as f:
                            metadata = json.load(f)
                    except Exception as e:
                        logger.error(f"Error loading metadata from {json_path}: {e}")
            
            # Process each chunk
            for i, chunk in enumerate(text_chunks):
                if not chunk.strip():
                    continue
                    
                # Generate unique document ID
                doc_id = self._generate_document_id(pdf_path, i)
                
                # Create document dictionary
                doc = {
                    "id": doc_id,
                    "text": chunk.strip(),
                    "metadata": {
                        **(metadata or {}),
                        "chunk_index": i,
                        "total_chunks": len(text_chunks),
                        "source": pdf_path,
                        "filename": os.path.basename(pdf_path),
                        "processed_at": datetime.now().isoformat(),
                    },
                    "distance": 0.0,  # Will be filled during search
                    "relevance_score": 0.0,  # Will be filled during search
                }
                
                documents.append(doc)
            
            # Mark as processed
            self.processed_pdfs.add(pdf_name)
            
            logger.info(f"Successfully processed {pdf_path} into {len(documents)} chunks")
            return documents
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            return []
    
    def _generate_document_id(self, pdf_path: str, chunk_index: int) -> str:
        """Generate a unique document ID"""
        content = f"{pdf_path}_{chunk_index}_{datetime.now().timestamp()}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def process_ministry_pdfs(self, ministry: str) -> List[Dict[str, Any]]:
        """
        Process all PDFs for a specific ministry
        Returns a list of all document chunks from the ministry's PDFs
        """
        try:
            logger.info(f"Processing PDFs for ministry: {ministry}")
            
            # Get ministry directory
            ministry_dir = Config.get_ministry_dir(ministry)
            
            if not ministry_dir.exists():
                logger.warning(f"Ministry directory does not exist: {ministry_dir}")
                return []
                
            # Get all PDFs in the ministry directory
            pdf_files = list(ministry_dir.glob("*.pdf"))
            
            if not pdf_files:
                logger.warning(f"No PDFs found for ministry: {ministry}")
                return []
                
            logger.info(f"Found {len(pdf_files)} PDFs for {ministry}")
            
            # Process each PDF
            all_documents = []
            
            for pdf_path in pdf_files:
                # Create basic metadata
                metadata = {
                    "ministry": ministry,
                    "date": "Unknown",
                    "session": str(Config.DEFAULT_SESSION),
                    "pdf_url": f"/data/ministry_pdfs/{Config.sanitize_ministry_name(ministry)}/{pdf_path.name}"
                }
                
                # Try to load more detailed metadata if available
                json_path = pdf_path.with_suffix('.json')
                if json_path.exists():
                    try:
                        with open(json_path, 'r') as f:
                            file_metadata = json.load(f)
                            # Update metadata with file-specific information
                            metadata.update(file_metadata)
                    except Exception as e:
                        logger.error(f"Error loading metadata from {json_path}: {e}")
                
                # Process the PDF
                documents = self.process_pdf(str(pdf_path), metadata)
                
                if documents:
                    all_documents.extend(documents)
            
            logger.info(f"Successfully processed {len(all_documents)} document chunks for {ministry}")
            return all_documents
            
        except Exception as e:
            logger.error(f"Error processing PDFs for ministry {ministry}: {e}")
            return []