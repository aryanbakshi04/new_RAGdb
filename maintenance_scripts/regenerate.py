from src.vector_store import VectorStore
vector_store = VectorStore()

vector_store._load_indexed_ministries()
vector_store._save_indexed_ministries()

print(f"Found {len(vector_store.indexed_ministries)} indexed ministries")