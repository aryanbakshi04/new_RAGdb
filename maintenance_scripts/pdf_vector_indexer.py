import os
from pathlib import Path
from tqdm import tqdm
import logging
import sys

# Add parent directory to path for src imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import Config
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("pdf_vector_indexer")

def main():
    print("Starting PDF vector embedding and classification process...")
    
    # Initialize processor and vector store
    doc_processor = DocumentProcessor()
    vector_store = VectorStore()

    # Get all PDFs in the cache directory
    pdf_dir = Path(Config.PDF_CACHE_DIR)
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {Config.PDF_CACHE_DIR}")
        return
    print(f"Found {len(pdf_files)} PDFs. Processing...")

    # Classify PDFs by ministry (assumes ministry name in filename or metadata)
    ministry_to_pdfs = {}
    for pdf_file in pdf_files:
        # Example: Ministry name in filename, e.g. Ministry_of_Finance_1234.pdf
        ministry = None
        for m in Config.MINISTRIES:
            if m.replace(" ", "_") in pdf_file.name:
                ministry = m
                break
        if not ministry:
            ministry = "Unknown"
        ministry_to_pdfs.setdefault(ministry, []).append(pdf_file)

    # Process and index PDFs by ministry
    total_chunks = 0
    for ministry, pdf_list in tqdm(ministry_to_pdfs.items(), desc="Indexing ministries"):
        print(f"\nProcessing {len(pdf_list)} PDFs for {ministry}...")
        documents = doc_processor.process_pdf_files(pdf_list, ministry=ministry)
        if not documents:
            logger.warning(f"No documents generated for {ministry}")
            continue
        vector_store.add_documents(documents, ministry=ministry)
        print(f"Added {len(documents)} chunks for {ministry}")
        total_chunks += len(documents)

    print(f"\nSuccessfully indexed {len(ministry_to_pdfs)} ministries with {total_chunks} total chunks.")
    print("Process complete! You can now run the Streamlit app.")

if __name__ == "__main__":
    main()
