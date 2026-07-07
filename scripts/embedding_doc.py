from sentence_transformers import SentenceTransformer
from typing import List, Any
import numpy as np

class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        
        try:
            self.model = SentenceTransformer(model_name)
            print(f"Initaialised model : {model_name}")
        except Exception as e:
            print(f"Either Model Not Available or something went wrong {model_name} : {e}")
            raise

    def embed_pdf(self, chunks: List[Any]) -> List[dict]:
        texts = [chunk.page_content for chunk in chunks]
        print(f"[INFO] Generating embeddings for {len(texts)} chunks...")
        try:
            embeddings = self.model.encode(texts, show_progress_bar=False)

            processed_data = []
            for i, chunk in enumerate(chunks):
                chunk_id = chunk.metadata.get("id", f"chunk_{i}")
                processed_data.append({
                    "id" : chunk_id,
                    "vector" : embeddings[i].tolist(),
                    "text" : chunk.page_content,
                    "metadata" : chunk.metadata,
                }) 
            return processed_data
        
        except Exception as e:
            print(f"Embedding Failed : {type(e).__name__}: {e}")
            return []
        
    def embed_query(self, query: str):
        try:
            vector = self.model.encode([query])
            return vector.tolist()
        except Exception as e:
            print(f"error with: {type(e).__name__}: {e}")
            raise
