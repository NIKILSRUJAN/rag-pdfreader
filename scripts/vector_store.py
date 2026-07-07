import chromadb

def get_collection(db_path: str = "./chroma_db", collection_name: str = "my_pdf_collection"):
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name=collection_name)
    return collection


def StoreInDB(processed_data, collection):

    if not processed_data:
        print("No processed data to store — skipping.")
        return

    ids = [item["id"] for item in processed_data]
    embeddings = [item["vector"] for item in processed_data]
    documents = [item["text"] for item in processed_data]
    metadatas = [item["metadata"] for item in processed_data]

    collection.add(
        ids = ids,
        embeddings = embeddings,
        documents = documents,
        metadatas = metadatas
    )

    print(f"successfully stored {len(ids)} chunks in chromaDB")

