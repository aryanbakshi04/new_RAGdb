from src.vector_store import VectorStore

# Initialize vector store
vector_store = VectorStore()

# Force regeneration from collection
vector_store._load_indexed_ministries()
vector_store._save_indexed_ministries()

# Verify
print(f"Found {len(vector_store.indexed_ministries)} indexed ministries")