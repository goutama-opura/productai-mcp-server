import faiss
import numpy as np
import os


class FaissIndex:
    def __init__(self, dim: int):
        """
        Initialize a FAISS index with specified vector dimension.
        Uses IndexFlatL2 (exact search with L2 distance).
        """
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.id_map = []  # map index IDs to data IDs or row indices


    @property
    def ntotal(self):
        """
        Expose the total number of vectors currently indexed.
        """
        return self.index.ntotal


    def add(self, vectors: np.ndarray, ids: list):
        """
        Add vectors and their associated IDs to the index.
        """
        if not isinstance(vectors, np.ndarray):
            raise ValueError("Vectors must be a numpy.ndarray")
        if vectors.shape[1] != self.dim:
            raise ValueError(f"Vectors dimensionality must be {self.dim}")

        self.index.add(vectors)
        self.id_map.extend(ids)


    def search(self, query_vector: np.ndarray, top_k: int = 5):
        """
        Search the index for top_k nearest neighbors of the query vector.
        Returns a list of tuples (id, distance).
        """
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)  # reshape single vector

        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for dist_row, idx_row in zip(distances, indices):
            res = []
            for dist, idx in zip(dist_row, idx_row):
                if idx < 0 or idx >= len(self.id_map):
                    continue
                res.append((self.id_map[idx], dist))
            results.append(res)
        return results


    def save(self, path: str):
        """
        Save the FAISS index and associated ID mapping.
        """
        faiss.write_index(self.index, path + '.index')

        # Save id_map
        with open(path + '.ids', 'w') as f:
            for _id in self.id_map:
                f.write(str(_id) + '\n')


    def load(self, path: str):
        """
        Load the FAISS index and associated ID mapping.
        """
        if not os.path.exists(path + '.index') or not os.path.exists(path + '.ids'):
            raise FileNotFoundError("Index or ID mapping file not found")

        self.index = faiss.read_index(path + '.index')
        with open(path + '.ids', 'r') as f:
            self.id_map = [line.strip() for line in f]

        if self.index.d != self.dim:
            raise ValueError(f"Loaded index dimension {self.index.d} != initialized dimension {self.dim}")
