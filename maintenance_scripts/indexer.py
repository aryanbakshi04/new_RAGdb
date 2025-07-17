import chromadb
from chromadb.config import Settings
from src.config import Config
import json
from pathlib import Path

# Initialize the database directly
settings = Settings(
    anonymized_telemetry=False,
    allow_reset=True,
    is_persistent=True,
    persist_directory=str(Config.VECTOR_DB_DIR),
)

client = chromadb.PersistentClient(path=str(Config.VECTOR_DB_DIR), settings=settings)
collection = client.get_collection("ministry_documents")

# Query unique ministries directly from the database
results = collection.get(include=["metadatas"], limit=1000000)
indexed_ministries = set()

# Extract unique ministry names
for metadata in results["metadatas"]:
    if metadata and "ministry" in metadata:
        indexed_ministries.add(metadata["ministry"])

# Save to JSON file
metadata_file = Path(Config.VECTOR_DB_DIR) / "indexed_ministries.json"
with open(metadata_file, "w") as f:
    json.dump({
        "ministries": list(indexed_ministries),
        "timestamp": "2025-07-14 23:30:00",
        "count": len(indexed_ministries)
    }, f, indent=2)

print(f"Fixed metadata file with {len(indexed_ministries)} ministries")
print("Ministries found:", sorted(indexed_ministries))