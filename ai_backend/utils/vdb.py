from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os
from config import VDB_PATH

embedding_fn = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vdb = Chroma(
    persist_directory=VDB_PATH,
    embedding_function=embedding_fn
)


def store_text(text: str, metadata: dict):
    vdb.add_texts([text], metadatas=[metadata])


def search_similar(text: str, k: int = 3):
    return vdb.similarity_search(text, k=k)
