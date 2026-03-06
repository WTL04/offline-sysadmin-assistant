from pymilvus.model.hybrid import BGEM3EmbeddingFunction
import numpy as np
import numpy.typing as npt
from typing import List, Dict, Tuple, Union
from pprint import pprint
from langchain_core.embeddings import Embeddings

class BGE_M3(Embeddings):
    """
    Unified BGE-M3 Wrapper. 
    Provides Dense embeddings for LangChain/Chroma and 
    Hybrid (Dense+Sparse) capabilities for custom RRF scoring.
    """

    def __init__(self, device: str = "cuda:0"):
        self.use_fp16 = True if device == "cuda:0" else False
        self.model = BGEM3EmbeddingFunction(
            model_name="BAAI/bge-m3", 
            device=device, 
            use_fp16=self.use_fp16
        )
        print(f"Model loaded: {device}, fp16: {self.use_fp16}")

    # --- 1. CORE LOGIC (The "Internal Engine") ---

    def _get_full_embeddings(self, texts: List[str]) -> Dict:
        """Internal helper to get both dense and sparse vectors."""
        # BGE-M3's encode_documents handles both
        return self.model.encode_documents(texts)

    # --- 2. LANGCHAIN INTERFACE (Chroma uses these) ---

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Standard LangChain method: returns only dense vectors as lists."""
        results = self._get_full_embeddings(texts)
        return results["dense"].tolist()

    def embed_query(self, text: str) -> List[float]:
        """Standard LangChain method: returns a single dense vector."""
        # Use encode_queries for better query-side performance
        results = self.model.encode_queries([text])
        return results["dense"][0].tolist()

    # --- 3. CUSTOM HYBRID METHODS (Your RRF Logic) ---

    def score_per_doc(self, query: str, docs: List[str], k: int = 60) -> List[Dict]:
        """
        Custom RRF scoring using both Dense and Sparse signals.
        """
        # Get hybrid embeddings for query and docs
        q_embeddings = self.model.encode_queries([query])
        d_embeddings = self._get_full_embeddings(docs)

        # Extract Dense
        q_dense = q_embeddings["dense"][0]
        d_denses = d_embeddings["dense"]

        # 1. Dense Ranking (Cosine Similarity)
        dense_sims = [self.cosine_similarity_numpy(q_dense, d) for d in d_denses]
        dense_ranks = np.argsort(dense_sims)[::-1]
        dense_lookup = {idx: rank + 1 for rank, idx in enumerate(dense_ranks)}

        # 2. Sparse Ranking (Dot Product)
        # q_sparse is a CSR matrix, we use matrix multiplication
        q_sparse = q_embeddings["sparse"][0]
        d_sparses = d_embeddings["sparse"]
        sparse_sims = (d_sparses @ q_sparse.T).toarray().flatten()
        sparse_ranks = np.argsort(sparse_sims)[::-1]
        sparse_lookup = {idx: rank + 1 for rank, idx in enumerate(sparse_ranks)}

        # 3. RRF Fusion
        results = []
        for i in range(len(docs)):
            d_r = dense_lookup.get(i, 0)
            s_r = sparse_lookup.get(i, 0)
            score = self.rrf_score(d_r, s_r, k)
            
            results.append({
                "doc": docs[i],
                "rrf_score": score,
                "dense_rank": d_r,
                "sparse_rank": s_r
            })

        return sorted(results, key=lambda x: x["rrf_score"], reverse=True)

    def cosine_similarity_numpy(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        dot = np.dot(vec1, vec2)
        norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
        return dot / (norm1 * norm2) if (norm1 > 0 and norm2 > 0) else 0.0

    def rrf_score(self, d_rank: int, s_rank: int, k: int) -> float:
        score = 0.0
        if d_rank > 0: score += 1.0 / (k + d_rank)
        if s_rank > 0: score += 1.0 / (k + s_rank)
        return score

# --- EXECUTION ---
if __name__ == "__main__":
    docs = ["AI research history", "Alan Turing's life", "Deep Learning basics"]
    query = "Who is Turing?"
    
    bge = BGE_M3(device="cpu") # Switch to cuda:0 for speed
    
    # Custom RRF Hybrid Search
    print("\n--- Custom RRF Results ---")
    pprint(bge.score_per_doc(query, docs))