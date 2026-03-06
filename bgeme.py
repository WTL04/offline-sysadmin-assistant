from pymilvus.model.hybrid import BGEM3EmbeddingFunction
import numpy as np
import numpy.typing as npt
from typing import List, Dict, Tuple
from pprint import pprint  # for pretty printing dictionaries


class BGE_M3:
    """
    Wrapper class for the BGEM3EmbeddingFunction to manage dense and sparse embeddings.
    """

    def __init__(self, device: str = "cuda:0"):
        """
        Initializes the BGE_M3 embedding model.

        Args:
            device (str): The computation device to use. Defaults to "cuda:0".
        """
        # initialize embedding model
        self.use_fp16 = True if device == "cuda:0" else False
        self.model = BGEM3EmbeddingFunction(
            model_name="BAAI/bge-m3", device=device, use_fp16=self.use_fp16
        )

    def embed(
        self, query: List[str], docs: List[str], verbose: bool = True
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """
        Generates embeddings for both the query and the provided documents.

        Args:
            query (List[str]): A list containing the query string(s).
            docs (List[str]): A list of document strings to be embedded.
            verbose (bool): If True, prints the dimensions of the generated embeddings. Defaults to True.

        Returns:
            Tuple[npt.NDArray, npt.NDArray]: A tuple containing the query embeddings and document embeddings dictionaries.
        """
        docs_embeddings = self.model.encode_documents(docs)
        query_embedding = self.model.encode_queries(query)

        if verbose:
            # Print dimension of dense embeddings
            print(
                "Dense document dim:",
                self.model.dim["dense"],
                docs_embeddings["dense"][0].shape,
            )
            # Since the sparse embeddings are in a 2D csr_array format, we convert them to a list for easier manipulation.
            print(
                "Sparse document dim:",
                self.model.dim["sparse"],
                list(docs_embeddings["sparse"])[0].shape,
            )

        return query_embedding, docs_embeddings

    def cosine_similarity_numpy(self, vec1, vec2):
        """
        Calculates the cosine similarity between two vectors.

        Args:
            vec1 (numpy.ndarray): The first input vector.
            vec2 (numpy.ndarray): The second input vector.

        Returns:
            float: The computed cosine similarity score. Returns 0 if either vector has a norm of zero.
        """
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0  # Handle zero vectors gracefully
        return dot_product / (norm_vec1 * norm_vec2)

    def rrf_score(self, dense_rank: int, sparse_rank: int, k: int = 60) -> float:
        """
        Calculates the Reciprocal Rank Fusion (RRF) score for a single document.

        Args:
            dense_rank (int): The document's rank based on dense embedding similarity.
            sparse_rank (int): The document's rank based on sparse embedding similarity.
            k (int): A smoothing constant to mitigate the impact of outliers in high ranks. Defaults to 60.

        Returns:
            float: The combined RRF score.
        """
        # score(doc) = 1 / (k + rank(dense docs)) + 1 / (k + rank(sparse docs))
        score = 0.0
        if dense_rank > 0:
            score += 1.0 / (k + dense_rank)
        if sparse_rank > 0:
            score += 1.0 / (k + sparse_rank)
        return score

    def score_per_doc(self, query: List[str], docs: List[str], k=60) -> List[Dict]:
        """
        Evaluates and ranks a list of documents against a query using hybrid search and RRF.

        Args:
            query (List[str]): A list containing the query string(s).
            docs (List[str]): A list of document strings to be scored.
            k (int): A smoothing constant for the RRF calculation. Defaults to 60.

        Returns:
            List[Dict]: A list of dictionaries containing the document text, RRF score, dense rank, and sparse rank, sorted in descending order by the RRF score.
        """
        query_embeddings, docs_embeddings = self.embed(query, docs)

        q_dense = query_embeddings["dense"][0]
        q_sparse = query_embeddings["sparse"][0]

        dense_similarities = []

        # computing similarity between dense
        for d_dense in docs_embeddings["dense"]:
            sim = self.cosine_similarity_numpy(q_dense, d_dense)
            dense_similarities.append(sim)

        # returns indices of sorted items in descending order
        dense_ranks = np.argsort(dense_similarities)[::-1]
        dense_rank_lookup = {idx: rank + 1 for rank, idx in enumerate(dense_ranks)}

        # sparse sims evaluated via Dot Product
        # q_sparse shape = 1 x V, docs_sparse shape = N x V, use matrix mul
        sparse_similarities = (
            (docs_embeddings["sparse"] @ q_sparse.T).toarray().flatten()
        )
        sparse_ranks = np.argsort(sparse_similarities)[::-1]
        sparse_rank_lookup = {idx: rank + 1 for rank, idx in enumerate(sparse_ranks)}

        # --- 3. Final RRF Scoring ---
        final_results = []
        for i in range(len(docs)):
            d_rank = dense_rank_lookup.get(i, 0)
            s_rank = sparse_rank_lookup.get(i, 0)

            combined_score = self.rrf_score(d_rank, s_rank, k)

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
    """
    Main execution function to demonstrate document ranking for a sample query.
    """
    docs = [
        "Artificial intelligence was founded as an academic discipline in 1956.",
        "Alan Turing was the first person to conduct substantial research in AI.",
        "Born in Maida Vale, London, Turing was raised in southern England.",
        "What is AI research",
    ]

    query = ["What is AI research"]
    model = BGE_M3()
    ranks = model.score_per_doc(query, docs)

    # pretty print
    pprint(ranks, indent=4)


if __name__ == "__main__":
    main()
