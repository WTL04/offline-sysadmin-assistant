from pymilvus.model.hybrid import BGEM3EmbeddingFunction
import numpy as np
from typing import List, Dict
from pprint import pprint, pformat  # for pretty printing dictionaries


# initialize embedding model
bge_m3_ef = BGEM3EmbeddingFunction(
    model_name="BAAI/bge-m3", device="cuda:0", use_fp16=True
)


def embed(query: List[str], docs: List[str]):
    docs_embeddings = bge_m3_ef.encode_documents(docs)
    query_embedding = bge_m3_ef.encode_queries(query)

    # Print embeddings
    print("Embeddings:", docs_embeddings)
    # Print dimension of dense embeddings
    print(
        "Dense document dim:", bge_m3_ef.dim["dense"], docs_embeddings["dense"][0].shape
    )
    # Since the sparse embeddings are in a 2D csr_array format, we convert them to a list for easier manipulation.
    print(
        "Sparse document dim:",
        bge_m3_ef.dim["sparse"],
        list(docs_embeddings["sparse"])[0].shape,
    )

    return query_embedding, docs_embeddings


def cosine_similarity_numpy(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0  # Handle zero vectors gracefully
    return dot_product / (norm_vec1 * norm_vec2)


def rrf_score(dense_rank: int, sparse_rank: int, k: int = 60) -> float:
    """
    Calculates RRF Score for a single document
    """
    # score(doc) = 1 / (k + rank(dense docs)) + 1 / (k + rank(sparse docs))
    score = 0.0
    if dense_rank > 0:
        score += 1.0 / (k + dense_rank)
    if sparse_rank > 0:
        score += 1.0 / (k + sparse_rank)
    return score


def score_per_doc(query: List[str], docs: List[str], k=60) -> List[Dict]:
    """
    Args:
        query: list of strings
        docs: list of strings
        k: int
    Returns:
        List of Dictionaries of RRF Scores in descending order
    """
    query_embeddings, docs_embeddings = embed(query, docs)

    q_dense = query_embeddings["dense"][0]
    q_sparse = query_embeddings["sparse"][0]

    dense_similarities = []

    # computing similarity between dense
    for d_dense in docs_embeddings["dense"]:
        sim = cosine_similarity_numpy(q_dense, d_dense)
        dense_similarities.append(sim)

    # returns indices of sorted items in descending order
    dense_ranks = np.argsort(dense_similarities)[::-1]
    dense_rank_lookup = {idx: rank + 1 for rank, idx in enumerate(dense_ranks)}

    # sparse sims evaluated via Dot Product
    # q_sparse shape = 1 x V, docs_sparse shape = N x V, use matrix mul
    sparse_similarities = (docs_embeddings["sparse"] @ q_sparse.T).toarray().flatten()
    sparse_ranks = np.argsort(sparse_similarities)[::-1]
    sparse_rank_lookup = {idx: rank + 1 for rank, idx in enumerate(sparse_ranks)}

    # --- 3. Final RRF Scoring ---
    final_results = []
    for i in range(len(docs)):
        d_rank = dense_rank_lookup.get(i, 0)
        s_rank = sparse_rank_lookup.get(i, 0)

        combined_score = rrf_score(d_rank, s_rank, k)

        final_results.append(
            {
                "doc": docs[i],
                "rrf_score": combined_score,
                "dense_rank": d_rank,
                "sparse_rank": s_rank,
            }
        )

    # Sort final results by RRF score descending
    return sorted(final_results, key=lambda x: x["rrf_score"], reverse=True)


def main():
    docs = [
        "Artificial intelligence was founded as an academic discipline in 1956.",
        "Alan Turing was the first person to conduct substantial research in AI.",
        "Born in Maida Vale, London, Turing was raised in southern England.",
        "What is AI research",
    ]

    query = ["What is AI research"]

    ranks = score_per_doc(query, docs)

    # pretty print
    pprint(ranks, indent=4)


if __name__ == "__main__":
    main()
