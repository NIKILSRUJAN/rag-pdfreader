import sys
from doc_loader_and_spilt import pdfLoader
from embedding_doc import EmbeddingManager
from vector_store import get_collection, StoreInDB
from retriver_and_context import retrieve, build_context
from LLM_connection import answer


def ingest(directory_path: str, embedder: EmbeddingManager, collection):
    print(f"[INGEST] Loading and splitting PDFs from '{directory_path}'...")
    chunks = pdfLoader(directory_path)

    if not chunks:
        print("[INGEST] No chunks produced — aborting ingestion.")
        return

    print(f"[INGEST] {len(chunks)} chunks created. Generating embeddings...")
    processed_data = embedder.embed_pdf(chunks)

    if not processed_data:
        print("[INGEST] Embedding failed — aborting ingestion.")
        return

    StoreInDB(processed_data, collection)
    print("[INGEST] Done.")


def query(user_query: str, embedder: EmbeddingManager, collection, k: int = 3):
    results = retrieve(user_query, collection, embedder, k)
    context = build_context(results)

    print("\n--- Retrieved Context ---")
    print(context)
    print("-------------------------\n")

    final_answer = answer(user_query, context)
    return final_answer


def main():
    embedder = EmbeddingManager(model_name="all-MiniLM-L6-v2")
    collection = get_collection()

    if len(sys.argv) < 2:
        print("Usage: python main.py [ingest|query]")
        return

    mode = sys.argv[1]

    if mode == "ingest":
        ingest("../data", embedder, collection)

    elif mode == "query":
        while True:
            user_query = input("Ask a question (or 'exit' to quit): ")
            if user_query.strip().lower() == "exit":
                break
            result = query(user_query, embedder, collection)
            print(f"\nAnswer: {result}\n")

    else:
        print(f"Unknown mode '{mode}'. Use 'ingest' or 'query'.")


if __name__ == "__main__":
    main()