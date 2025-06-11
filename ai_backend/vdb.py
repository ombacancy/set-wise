from datetime import datetime

import chromadb
from chromadb.utils import embedding_functions
from config import settings

chroma_client = chromadb.Client()

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=settings.EMBED_MODEL
)

db_collection = chroma_client.get_or_create_collection(
    name=settings.VDB_COLLECTION,
    embedding_function=embedding_fn
)


def store_log(user_id: str, text: str, metadata: dict):
    db_collection.add(
        documents=[text],
        ids=[f"{user_id}_{datetime.utcnow().isoformat()}"],
        metadatas=[metadata]
    )


def query_logs(user_id: str, query_text: str, k=5):
    return db_collection.query(
        query_texts=[query_text],
        n_results=k,
        where={"user_id": user_id}
    )
