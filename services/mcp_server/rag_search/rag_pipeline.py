import numpy as np
from services.mcp_server.rag_search.ingestion import load_csv
from services.mcp_server.rag_search.text_embedding import generate_text_embedding
from services.mcp_server.rag_search.image_embedding import generate_image_embedding, fetch_image
from services.mcp_server.rag_search.insert_vectors import VectorStore
from services.mcp_server.rag_search.retriever import Retriever
from services.mcp_server.rag_search.faiss_index import FaissIndex
from services.mcp_server.rag_search.generator import generate_answer

class RagPipeline:
    def __init__(self, data_path='data/raw.csv'):
        # Step 1: Load raw data
        self.data = load_csv(data_path)
        if not self.data:
            raise ValueError("No data loaded for RAG pipeline.")
        
        # Step 2: Generate embeddings for each data chunk
        self.embeddings, self.ids, self.metadata = self._generate_embeddings(self.data)
        
        # Step 3: Initialize vector store and load embeddings
        dim = self.embeddings.shape[1]
        self.vector_store = VectorStore(dim)
        self.vector_store.add_vectors(self.embeddings, self.ids)
        
        # Step 4: Setup retriever with vector store and data
        self.retriever = Retriever(self.vector_store, self.data)

    def _generate_embeddings(self, data):
        embeddings = []
        ids = []
        metadata = []
        for idx, item in enumerate(data):
            try:
                if 'image_url' in item and item['image_url']:
                    img = fetch_image(item['image_url'])
                    emb = generate_image_embedding(img)
                else:
                    emb = generate_text_embedding(item.get('text', ''))
                embeddings.append(emb)
                ids.append(str(idx))
                metadata.append(item)
            except Exception as e:
                print(f"Embedding error at index {idx}: {e}")
        return np.array(embeddings), ids, metadata

    def run_query(self, query: str):
        # Perform retrieval
        results = self.retriever.hybrid_search(query, top_k=5)
        # Generate answer
        answer = generate_answer(query, results)
        return answer, results

# Usage example:
# pipeline = RagPipeline()
# answer, sources = pipeline.run_query("What is FAISS?")
