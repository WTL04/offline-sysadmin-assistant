import torch.nn.functional as F
import faiss
from pymilvus.model.hybrid import BGEM3EmbeddingFunction


bge_m3_ef = BGEM3EmbeddingFunction(
    model_name="BAAI/bge-m3",  # Specify the model name
    device="cuda:0",  # Specify the device to use, e.g., 'cpu' or 'cuda:0'
    use_fp16=True,  # Specify whether to use fp16. Set to `False` if `device` is `cpu`.
)

docs = [
    "Artificial intelligence was founded as an academic discipline in 1956.",
    "Alan Turing was the first person to conduct substantial research in AI.",
    "Born in Maida Vale, London, Turing was raised in southern England.",
]

query = ["What is AI research"]

docs_embeddings = bge_m3_ef.encode_documents(docs)
query_embedding = bge_m3_ef.encode_queries(query)

# Print embeddings
print("Embeddings:", docs_embeddings)
# Print dimension of dense embeddings
print("Dense document dim:", bge_m3_ef.dim["dense"], docs_embeddings["dense"][0].shape)
# Since the sparse embeddings are in a 2D csr_array format, we convert them to a list for easier manipulation.
print(
    "Sparse document dim:",
    bge_m3_ef.dim["sparse"],
    list(docs_embeddings["sparse"])[0].shape,
)

print(type(docs_embeddings["dense"][0]))  # numpy array
print(type(bge_m3_ef.dim["dense"]))  # int


# TODO: implement reciprocal rank fusion (rrf)
def score(k, doc):
    # score(doc) = 1 / (k + rank(dense docs)) + 1/ (k + rank(sparse docs)) 

