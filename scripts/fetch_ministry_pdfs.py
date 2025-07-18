# PDF fetcher to download ALL PDFs

import os
import asyncio
import logging
import sys
from pathlib import Path
from tqdm import tqdm
import time
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import Config
from src.sansad_client import SansadClient
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("pdf_fetcher.log"), logging.StreamHandler()],
)
logger = logging.getLogger("pdf_fetcher")


class ComprehensivePDFFetcher:
    def __init__(self):

        self.sansad_client = SansadClient()
        self.doc_processor = DocumentProcessor()
        self.vector_store = VectorStore()

        self.stats = {
            "total_pdfs_found": 0,
            "total_pdfs_downloaded": 0,
            "ministries_processed": 0,
            "errors": 0,
        }

    async def fetch_pdfs_for_ministry(self, ministry):
        logger.info(f"Fetching PDFs for {ministry}")

        downloaded_count = 0
        page = 1
        max_pages = 625
        has_more_pages = True

        while has_more_pages and page <= max_pages:
            try:
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

                for question in tqdm(
                    questions, desc=f"{ministry} page {page}", leave=False
                ):
                    pdf_url = question.get("pdf_url")
                    if not pdf_url:
                        continue

                    pdf_path = await self.sansad_client.download_pdf(pdf_url)

                    if pdf_path:
                        downloaded_count += 1
                        self.stats["total_pdfs_downloaded"] += 1
                page += 1
                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"Error fetching PDFs for {ministry} on page {page}: {e}")
                self.stats["errors"] += 1
                page += 1

        self.stats["ministries_processed"] += 1
        logger.info(f"Downloaded {downloaded_count} PDFs for {ministry}")
        return downloaded_count

    async def fetch_all_ministries(self):
        start_time = time.time()

        print(f"Starting COMPREHENSIVE PDF fetch process at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        print(f"Target: ALL available PDFs for ALL ministries")
        print("This process may take several hours to complete.")
        print("=" * 70)

        os.makedirs(Config.PDF_CACHE_DIR, exist_ok=True)

        for ministry in tqdm(Config.MINISTRIES, desc="Processing ministries"):
            count = await self.fetch_pdfs_for_ministry(ministry)
            print(f"Downloaded {count} PDFs for {ministry}")

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
        print("\nClassifying and organizing PDFs...")
        pdf_files = list(Path(Config.PDF_CACHE_DIR).glob("*.pdf"))

        if not pdf_files:
            print(f"No PDFs found in {Config.PDF_CACHE_DIR}")
            return

        print(f"Found {len(pdf_files)} PDFs to classify")

    async def build_vector_database(self):
        print("\nBuilding vector database...")

        self.vector_store.clear()
        total_chunks = 0
        indexed_ministries = 0

        for ministry in tqdm(Config.MINISTRIES, desc="Indexing ministries"):
            try:
                print(f"\nProcessing {ministry}...")
                documents = self.doc_processor.process_ministry_pdfs(ministry)

                if not documents:
                    logger.warning(f"No documents generated for {ministry}")
                    continue

                self.vector_store.add_documents(documents, ministry=ministry)
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
    Config.setup_directories()

    fetcher = ComprehensivePDFFetcher()
    await fetcher.fetch_all_ministries()
    await fetcher.classify_and_organize_pdfs()
    await fetcher.build_vector_database()

    print("\nProcess complete! You can now run the App")


if __name__ == "__main__":
    asyncio.run(main())
