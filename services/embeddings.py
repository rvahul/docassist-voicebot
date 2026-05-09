from sentence_transformers import SentenceTransformer

# Downloads ~90MB on first run, then cached forever
model = SentenceTransformer('all-MiniLM-L6-v2')

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def get_embedding(text: str) -> list[float]:
    """Generate embedding for a single text."""
    return model.encode(text).tolist()


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for multiple texts."""
    return [model.encode(t).tolist() for t in texts]