import os
import logging
import asyncio
from datetime import datetime
import time
from pathlib import Path
from tqdm import tqdm
import sys
import json
import shutil
from src.config import Config
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("ministry_database.log"), logging.StreamHandler()],
)
logger = logging.getLogger("ministry_database")


class MinistryDatabaseCreator:

    def __init__(self, force_rebuild=False):
        self.force_rebuild = force_rebuild
        self.doc_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        self.stats = {
            "total_pdfs": 0,
            "classified_pdfs": 0,
            "unknown_pdfs": 0,
            "ministries_with_pdfs": 0,
            "total_chunks": 0,
            "errors": 0,
        }

        self.ministry_counts = {ministry: 0 for ministry in Config.MINISTRIES}
        self.ministry_counts["Unknown Ministry"] = 0

    async def create_database(self):
        start_time = time.time()

        print("=" * 70)
        print(f"MINISTRY DATABASE CREATOR - Started at {datetime.now().isoformat()}")
        print("=" * 70)

        print("\nStep 1: Creating directory structure...")
        Config.setup_directories()

        print("\nStep 2: Classifying and organizing PDFs...")
        await self.classify_pdfs()

        print("\nStep 3: Verifying ministry PDF distribution...")
        self.verify_distribution()

        print("\nStep 4: Building vector database...")
        await self.build_vector_database()

        elapsed_time = time.time() - start_time

        print("\nOperation Complete!")
        print("=" * 70)
        print(f"Total PDFs processed: {self.stats['total_pdfs']}")
        print(f"PDFs successfully classified: {self.stats['classified_pdfs']}")
        print(f"PDFs marked as unknown: {self.stats['unknown_pdfs']}")
        print(
            f"Ministries with PDFs: {self.stats['ministries_with_pdfs']}/{len(Config.MINISTRIES)}"
        )
        print(f"Total chunks indexed: {self.stats['total_chunks']}")
        print(f"Errors encountered: {self.stats['errors']}")
        print(f"Total runtime: {elapsed_time:.1f} seconds")
        print("=" * 70)

    async def classify_pdfs(self):
        pdf_files = list(Path(Config.PDF_CACHE_DIR).glob("*.pdf"))

        if not pdf_files:
            print(f"No PDFs found in {Config.PDF_CACHE_DIR}")
            return

        self.stats["total_pdfs"] = len(pdf_files)
        print(f"Found {len(pdf_files)} PDFs in cache directory")

        print("\nPDF Distribution by Ministry:")
        print("-" * 50)

        ministries_with_pdfs = 0
        for ministry, count in sorted(
            self.ministry_counts.items(), key=lambda x: x[1], reverse=True
        ):
            if count > 0:
                print(f"{ministry}: {count} PDFs")
                if ministry != "Unknown Ministry":
                    ministries_with_pdfs += 1

        self.stats["ministries_with_pdfs"] = ministries_with_pdfs

    def verify_distribution(self):
        """Verify each ministry has PDFs and redistribute if needed"""
        empty_ministries = [
            m
            for m, c in self.ministry_counts.items()
            if c == 0 and m != "Unknown Ministry"
        ]

        if not empty_ministries:
            print("All ministries have PDFs - no redistribution needed")
            return

        print(
            f"Found {len(empty_ministries)} ministries with no PDFs - redistributing..."
        )

        self._redistribute_pdfs(empty_ministries)
        self.stats["ministries_with_pdfs"] = len(Config.MINISTRIES) - len(
            [
                m
                for m, c in self.ministry_counts.items()
                if c == 0 and m != "Unknown Ministry"
            ]
        )

    def _redistribute_pdfs(self, empty_ministries):
        donor_ministries = [
            m
            for m, c in self.ministry_counts.items()
            if c > 10 and m != "Unknown Ministry"
        ]

        if not donor_ministries:
            print("No ministries have excess PDFs to redistribute")
            return

        for empty_ministry in empty_ministries:
            empty_dir = Config.get_ministry_dir(empty_ministry)

            donor = max(donor_ministries, key=lambda m: self.ministry_counts[m])
            donor_dir = Config.get_ministry_dir(donor)
            donor_pdfs = list(donor_dir.glob("*.pdf"))
            if not donor_pdfs:
                continue

            pdfs_to_move = donor_pdfs[: min(3, len(donor_pdfs))]

            for pdf_path in pdfs_to_move:
                try:
                    target_path = empty_dir / pdf_path.name
                    shutil.copy2(pdf_path, target_path)

                    metadata = {
                        "ministry": empty_ministry,
                        "original_ministry": donor,
                        "confidence": 0.1,
                        "original_path": str(pdf_path),
                        "classified_at": Config.CURRENT_TIME,
                        "classification_method": "redistribution",
                    }

                    metadata_path = target_path.with_suffix(".json")
                    with open(metadata_path, "w") as f:
                        json.dump(metadata, f, indent=2)

                    print(
                        f"Redistributed {pdf_path.name} from {donor} to {empty_ministry}"
                    )

                    self.ministry_counts[donor] -= 1
                    self.ministry_counts[empty_ministry] += 1

                except Exception as e:
                    logger.error(f"Error redistributing {pdf_path}: {e}")
                    self.stats["errors"] += 1

    async def build_vector_database(self):
        if self.force_rebuild and self.vector_store.indexed_ministries:
            confirm = input("WARNING: This will delete ALL indexed data. Type 'YES' to continue: ")
            if confirm != "YES":
                print("Aborted.")
                return
            print("Clearing existing vector store...")
            self.vector_store.clear()

        total_chunks = 0
        indexed_ministries = 0

        for ministry in tqdm(Config.MINISTRIES, desc="Indexing ministries"):
            try:
                if not self.force_rebuild and self.vector_store.is_ministry_indexed(
                    ministry
                ):
                    print(f"Ministry {ministry} already indexed - skipping")
                    indexed_ministries += 1
                    continue

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

        self.stats["total_chunks"] = total_chunks
        print(
            f"\nSuccessfully indexed {indexed_ministries} ministries with {total_chunks} total chunks"
        )


if __name__ == "__main__":

    force_rebuild = "--force" in sys.argv
    creator = MinistryDatabaseCreator(force_rebuild=force_rebuild)
    asyncio.run(creator.create_database())