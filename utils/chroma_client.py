import chromadb

_client = chromadb.PersistentClient(path="./chroma_store")

def get_client():
    return _client

def get_or_create_collection(name: str):
    return _client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"}
    )

def delete_collection(name: str):
    try:
        _client.delete_collection(name)
    except Exception:
        pass
