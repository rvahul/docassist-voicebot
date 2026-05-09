import faiss
import numpy as np
import json
import os

# In-memory store: { userId_documentId: { "index": faiss_index, "chunks": [...] } }
vector_store: dict = {}


def store_embeddings(user_id: str, document_id: str, chunks: list[str], embeddings: list[list[float]]):
    """Store chunk embeddings in FAISS index for a specific user+document."""
    key = f"{user_id}_{document_id}"
    dim = len(embeddings[0])

    # Create FAISS index with cosine similarity (using L2 on normalized vectors)
    index = faiss.IndexFlatIP(dim)  # Inner Product = cosine sim when normalized

    vectors = np.array(embeddings).astype("float32")
    faiss.normalize_L2(vectors)    # normalize for cosine similarity
    index.add(vectors)

    vector_store[key] = {
        "index": index,
        "chunks": chunks
    }


def retrieve_top_k(user_id: str, document_id: str, query_embedding: list[float], k: int = 3) -> list[dict]:
    """Retrieve top-K most similar chunks for a query."""
    key = f"{user_id}_{document_id}"

    if key not in vector_store:
        return []

    store = vector_store[key]
    index = store["index"]
    chunks = store["chunks"]

    query_vec = np.array([query_embedding]).astype("float32")
    faiss.normalize_L2(query_vec)

    scores, indices = index.search(query_vec, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append({
            "chunk": chunks[idx],
            "score": float(score)
        })

    return results
