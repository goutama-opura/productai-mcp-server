import numpy as np
from src.vector_store.faiss_index import FaissIndex


class VectorStore:
    def __init__(self, dim: int):
        self.faiss_index = FaissIndex(dim)
        self.ids = []

    def add_vectors(self, embeddings, ids: list):
        try:
            # Convert to np.ndarray if needed and ensure float32
            if not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings, dtype=np.float32)
            else:
                embeddings = embeddings.astype(np.float32)

            # Ensure embeddings shape is 2D
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)

            self.faiss_index.add(embeddings, ids)
            self.ids.extend(ids)
        except Exception as e:
            print(f"Error adding vectors to FAISS index: {e}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        try:
            if not isinstance(query_embedding, np.ndarray):
                query_embedding = np.array(query_embedding, dtype=np.float32)

            # Ensure query embedding is 2D
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)

            # Check if faiss_index has vectors
            if self.faiss_index.ntotal == 0:
                print("FAISS index is empty. No vectors to search.")
                return None, None

            distances, indices = self.faiss_index.search(query_embedding, top_k)
            return distances, indices
        except Exception as e:
            print(f"Error searching FAISS index: {e}")
            return None, None
