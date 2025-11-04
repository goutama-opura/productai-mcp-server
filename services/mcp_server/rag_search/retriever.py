import numpy as np

class Retriever:
    def __init__(self, faiss_index, dataframe):
        self.faiss_index = faiss_index  # This should be VectorStore or FaissIndex wrapper instance
        self.df = dataframe

    def search(self, query_embedding: np.ndarray, top_k=5):
        # Access ntotal from the internal faiss_index inside VectorStore
        if self.faiss_index.faiss_index.ntotal == 0:
            print("FAISS index is empty, no vectors to search.")
            return [], []

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        try:
            raw_results = self.faiss_index.search(query_embedding, top_k)
            distances = []
            indices = []
            for res in raw_results:
                if res:
                    id_list, dist_list = zip(*res)
                else:
                    id_list, dist_list = [], []
                indices.append(id_list)
                distances.append(dist_list)
            return distances, indices
        except Exception as e:
            print(f"Error during FAISS search: {e}")
            return [], []


    def hybrid_search(self, query_embedding: np.ndarray, top_k=5):
        distances, indices = self.search(query_embedding, top_k)
        enriched_results = []

        # FAISS search returns batched results, so take first batch
        if not distances or not indices:
            return enriched_results

        for doc_id, distance in zip(indices[0], distances[0]):
            if doc_id is not None and doc_id < len(self.df):
                record = self.df.iloc[doc_id]
                enriched_results.append({
                    "product_id": doc_id,
                    "title": record.get('Title', ''),
                    "vendor": record.get('Vendor', ''),
                    "distance": distance
                })
        return enriched_results
