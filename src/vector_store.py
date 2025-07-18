import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import os
import json
from pathlib import Path
from .config import Config

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self):
        self._initialize_db()
        self.indexed_ministries = set()
        self._load_indexed_ministries()

    def _initialize_db(self):
        try:
            settings = Settings(
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=True,
                persist_directory=str(Config.VECTOR_DB_DIR),
            )

            self.client = chromadb.PersistentClient(
                path=str(Config.VECTOR_DB_DIR), settings=settings
            )

            self.embedding_function = (
                embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=Config.EMBEDDING_MODEL
                )
            )

            self.collection = self.client.get_or_create_collection(
                name="ministry_documents",
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"},
            )

            logger.info("Successfully initialized vector database")

        except Exception as e:
            logger.error(f"Error initializing vector database: {e}")
            raise

    def _load_indexed_ministries(self):
        try:
            metadata_path = Path(Config.VECTOR_DB_DIR) / "indexed_ministries.json"

            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    data = json.load(f)
                    if "ministries" in data:
                        self.indexed_ministries = set(data["ministries"])

                logger.info(
                    f"Loaded {len(self.indexed_ministries)} indexed ministries from metadata"
                )
                return

            try:
                results = self.collection.get()

                if not results or not results.get("metadatas"):
                    logger.info("No documents found in vector store")
                    return

                for metadata in results["metadatas"]:
                    if metadata and "ministry" in metadata:
                        self.indexed_ministries.add(metadata["ministry"])

                self._save_indexed_ministries()

                logger.info(
                    f"Found {len(self.indexed_ministries)} indexed ministries from collection"
                )

            except Exception as e:
                logger.warning(f"Error checking collection for ministries: {e}")

        except Exception as e:
            logger.warning(f"Error loading indexed ministries: {e}")

    def _save_indexed_ministries(self):
        try:
            metadata_path = Path(Config.VECTOR_DB_DIR) / "indexed_ministries.json"

            with open(metadata_path, "w") as f:
                json.dump(
                    {
                        "ministries": list(self.indexed_ministries),
                        "updated_at": datetime.now().isoformat(),
                        "updated_by": Config.CURRENT_USER,
                    },
                    f,
                    indent=2,
                )

            logger.info(
                f"Saved {len(self.indexed_ministries)} indexed ministries to metadata"
            )

        except Exception as e:
            logger.warning(f"Error saving indexed ministries: {e}")

    def is_ministry_indexed(self, ministry: str) -> bool:
        return ministry in self.indexed_ministries

    def add_ministry_to_indexed(self, ministry: str):
        self.indexed_ministries.add(ministry)
        self._save_indexed_ministries()

    def create_embedding(self, text: str) -> List[float]:
        try:
            return self.embedding_function([text])[0]
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            raise

    def add_documents(self, documents: List[Dict[str, Any]], ministry: str = None):
        try:
            if not documents:
                logger.warning("No documents to add")
                return

            batch_size = 100
            total_added = 0

            for i in range(0, len(documents), batch_size):
                batch = documents[i : i + batch_size]

                ids = []
                texts = []
                metadatas = []

                for doc in batch:
                    if not isinstance(doc, dict) or "text" not in doc:
                        continue

                    text = doc["text"].strip()
                    if not text:
                        continue

                    metadata = self._clean_metadata(doc.get("metadata", {}))

                    if ministry and "ministry" not in metadata:
                        metadata["ministry"] = ministry

                    doc_id = doc.get("id", f"doc_{int(time.time())}_{i}_{len(ids)}")

                    ids.append(doc_id)
                    texts.append(text)
                    metadatas.append(metadata)

                if texts:
                    self.collection.add(ids=ids, documents=texts, metadatas=metadatas)
                    total_added += len(texts)

            logger.info(f"Added {total_added} documents to vector store")

            if ministry:
                self.add_ministry_to_indexed(ministry)

        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise

    def _clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        cleaned = {}

        for key, value in metadata.items():
            if value is None:
                continue

            if isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            else:
                cleaned[key] = str(value)

        if "ministry" not in cleaned:
            cleaned["ministry"] = "Unknown Ministry"

        return cleaned

    def search_with_embedding(
        self, embedding: List[float], ministry: str, n_results: int = 10
    ) -> List[Dict[str, Any]]:
        try:
            try:
                where_clause = {"ministry": {"$eq": ministry}} if ministry else None

                results = self.collection.query(
                    query_embeddings=[embedding],
                    n_results=n_results * 2,
                    where=where_clause,
                )

                if results["ids"] and results["ids"][0]:
                    return self._process_search_results(results, n_results)

            except Exception as inner_e:
                logger.warning(f"Error searching with ministry filter: {inner_e}")

            logger.info(f"Trying fallback search without ministry filter for {ministry}")
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=n_results * 2
            )

            return self._process_search_results(results, n_results)

        except Exception as e:
            logger.error(f"Error searching with embedding: {e}")
            return []

    def _process_search_results(self, results, n_results):
        if not results["ids"] or not results["ids"][0]:
            return []

        documents = []
        seen_texts = set()  

        for i in range(len(results["ids"][0])):
            doc_id = results["ids"][0][i]
            doc_text = results["documents"][0][i]
            doc_metadata = results["metadatas"][0][i]

            if doc_text in seen_texts:
                continue

            seen_texts.add(doc_text)

            distance = results["distances"][0][i] if "distances" in results else 0.0
            similarity = 1.0 - distance 

            document = {
                "id": doc_id,
                "text": doc_text,
                "metadata": doc_metadata,
                "distance": distance,
                "relevance_score": similarity,
            }

            documents.append(document)

            if len(documents) >= n_results:
                break

        documents.sort(key=lambda x: x["relevance_score"], reverse=True)

        return documents

    def search_by_text(
        self, query: str, ministry: str, n_results: int = 10
    ) -> List[Dict[str, Any]]:
        try:
            embedding = self.create_embedding(query)

            return self.search_with_embedding(embedding, ministry, n_results)

        except Exception as e:
            logger.error(f"Error searching by text: {e}")
            return []

    def clear(self):
        try:
            self.collection.delete(where={})
            self.indexed_ministries.clear()
            self._save_indexed_ministries()
            logger.info("Cleared vector store")
        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")