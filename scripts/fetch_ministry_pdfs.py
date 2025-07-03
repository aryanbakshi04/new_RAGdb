# PDF fetcher to download ALL PDFs

import os
import asyncio
import logging
import sys
from pathlib import Path
from tqdm import tqdm
import time

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import Config
from src.sansad_client import SansadClient
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("pdf_fetcher.log"), logging.StreamHandler()],
)
logger = logging.getLogger("pdf_fetcher")


class ComprehensivePDFFetcher:
    """Fetches ALL PDFs from Sansad.in API for each ministry"""

    def __init__(self):

        self.sansad_client = SansadClient()
        self.doc_processor = DocumentProcessor()
        self.vector_store = VectorStore()

        # Stats
        self.stats = {
            "total_pdfs_found": 0,
            "total_pdfs_downloaded": 0,
            "ministries_processed": 0,
            "errors": 0,
        }

    async def fetch_pdfs_for_ministry(self, ministry):
        """Fetch ALL PDFs for a specific ministry"""
        logger.info(f"Fetching PDFs for {ministry}")

        # Track PDFs downloaded for this ministry
        downloaded_count = 0
        page = 1
        max_pages = 625  # High limit to get all available pages
        has_more_pages = True

        # Process until we've reached the end of results or max page limit
        while has_more_pages and page <= max_pages:
            try:
                # Fetch questions for this ministry
                questions = await self.sansad_client.fetch_questions(
                    ministry=ministry, page=page
                )

                if not questions:
                    logger.info(
                        f"No more questions found for {ministry} after page {page-1}"
                    )
                    has_more_pages = False
                    break

                logger.info(
                    f"Found {len(questions)} questions for {ministry} on page {page}"
                )
                self.stats["total_pdfs_found"] += len(questions)

                # Download PDFs for ALL questions on this page
                for question in tqdm(
                    questions, desc=f"{ministry} page {page}", leave=False
                ):
                    pdf_url = question.get("pdf_url")
                    if not pdf_url:
                        continue

                    # Download PDF
                    pdf_path = await self.sansad_client.download_pdf(pdf_url)

                    if pdf_path:
                        downloaded_count += 1
                        self.stats["total_pdfs_downloaded"] += 1

                # Continue to next page
                page += 1

                # Small delay to avoid rate limiting
                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"Error fetching PDFs for {ministry} on page {page}: {e}")
                self.stats["errors"] += 1
                # Continue to next page despite errors
                page += 1

        self.stats["ministries_processed"] += 1
        logger.info(f"Downloaded {downloaded_count} PDFs for {ministry}")
        return downloaded_count

    async def fetch_all_ministries(self):
        """Fetch ALL PDFs for ALL ministries"""
        start_time = time.time()

        print(f"Starting COMPREHENSIVE PDF fetch process at {Config.CURRENT_TIME}")
        print(f"User: {Config.CURRENT_USER}")
        print("=" * 70)
        print(f"Target: ALL available PDFs for ALL ministries")
        print("This process may take several hours to complete.")
        print("=" * 70)

        # Create PDF cache directory
        os.makedirs(Config.PDF_CACHE_DIR, exist_ok=True)

        # Process ALL ministries (not just the first 5)
        for ministry in tqdm(Config.MINISTRIES, desc="Processing ministries"):
            count = await self.fetch_pdfs_for_ministry(ministry)
            print(f"Downloaded {count} PDFs for {ministry}")

        # Print statistics
        elapsed_time = time.time() - start_time

        print("\nFetching complete!")
        print("=" * 70)
        print(f"Total PDFs found: {self.stats['total_pdfs_found']}")
        print(f"Total PDFs downloaded: {self.stats['total_pdfs_downloaded']}")
        print(
            f"Ministries processed: {self.stats['ministries_processed']}/{len(Config.MINISTRIES)}"
        )
        print(f"Errors encountered: {self.stats['errors']}")
        print(f"Total fetch time: {elapsed_time:.1f} seconds")
        print("=" * 70)

    async def classify_and_organize_pdfs(self):
        """Classify all downloaded PDFs and organize by ministry"""
        print("\nClassifying and organizing PDFs...")

        # Get all PDFs from cache directory
        pdf_files = list(Path(Config.PDF_CACHE_DIR).glob("*.pdf"))

        if not pdf_files:
            print(f"No PDFs found in {Config.PDF_CACHE_DIR}")
            return

        print(f"Found {len(pdf_files)} PDFs to classify")

    async def build_vector_database(self):
        """Build vector database from organized PDFs"""
        print("\nBuilding vector database...")

        # Clear existing vector store
        self.vector_store.clear()

        # Process each ministry
        total_chunks = 0
        indexed_ministries = 0

        for ministry in tqdm(Config.MINISTRIES, desc="Indexing ministries"):
            try:
                # Process ministry PDFs
                print(f"\nProcessing {ministry}...")
                documents = self.doc_processor.process_ministry_pdfs(ministry)

                if not documents:
                    logger.warning(f"No documents generated for {ministry}")
                    continue

                # Add to vector store
                self.vector_store.add_documents(documents, ministry=ministry)

                # Update counts
                total_chunks += len(documents)
                indexed_ministries += 1

                print(f"Added {len(documents)} chunks for {ministry}")

            except Exception as e:
                logger.error(f"Error processing {ministry}: {e}")
                self.stats["errors"] += 1

        print(
            f"\nSuccessfully indexed {indexed_ministries} ministries with {total_chunks} total chunks"
        )


async def main():
    # Create directories
    Config.setup_directories()

    # Create fetcher
    fetcher = ComprehensivePDFFetcher()

    # Fetch ALL PDFs
    await fetcher.fetch_all_ministries()

    # Classify and organize PDFs
    await fetcher.classify_and_organize_pdfs()

    # Build vector database
    await fetcher.build_vector_database()

    print("\nProcess complete! You can now run the Streamlit app.")


if __name__ == "__main__":
    asyncio.run(main())
