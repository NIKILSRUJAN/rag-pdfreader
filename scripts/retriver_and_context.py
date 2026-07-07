from embedding_doc import EmbeddingManager


def retrieve(query: str, collection, embedder, k: int):
    query_vector = embedder.embed_query(query)

    results = collection.query(
        query_embeddings = query_vector,
        n_results = k
    )

    return results

def build_context(results) -> str:
    documents = results.get("documents", [[]])[0] #[0] means query I used is indexed at 0
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not documents:
        return "No relevant context found."

    pieces = []
    for i, doc_text in enumerate(documents):
        meta = metadatas[i] if i < len(metadatas) else {}
        source = meta.get("source", "unknown source")
        page = meta.get("page", "unknown page")
        distance = distances[i] if i < len(distances) else None

        source_label = f"[Source: {source}, page {page}"
        if distance is not None:
            source_label += f", distance: {distance:.4f}"
        source_label += "]"

        piece = f"{source_label}\n{doc_text}"
        pieces.append(piece)

    full_context = "\n\n---\n\n".join(pieces)
    return full_context
