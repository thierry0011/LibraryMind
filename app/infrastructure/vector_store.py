import chromadb
from pathlib import Path

from app.exceptions import VectorStoreException

_DB_PATH = str(Path(__file__).parent.parent.parent / "books_chroma_db")


class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=_DB_PATH)
        try:
            self.collection = self.client.get_or_create_collection(
                name="books",
                metadata={
                    "hnsw:space": "cosine",
                    "hnsw:ef_construction": 200,
                    "hnsw:M": 16,
                },
            )
        except Exception:
            # ChromaDB >=1.x Rust backend does not accept hnsw params in metadata;
            # fall back to cosine-only so the server starts regardless of version.
            self.collection = self.client.get_or_create_collection(
                name="books",
                metadata={"hnsw:space": "cosine"},
            )

    def upsert_books(self, id: str, embedding: list, metadata: dict, document: str):
        """
        Upsert a book into the vector store.
        """
        try:
            self.collection.upsert(
                ids=[id],
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[document],
            )
        except Exception as exc:
            raise VectorStoreException(f"ChromaDB upsert failed: {exc}") from exc

    def search_books(self, query_embedding: list, top_k: int = 5) -> list:
        """
        Search for books in the vector store.
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["metadatas", "documents", "distances"],
            )
        except Exception as exc:
            raise VectorStoreException(f"ChromaDB query failed: {exc}") from exc

        ids = results["ids"][0] if results["ids"] else []
        documents = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else []

        books = []
        for i in range(len(ids)):
            books.append(
                {
                    "id": ids[i],
                    "document": documents[i],
                    "metadata": metadatas[i],
                    # ChromaDB returns cosine distance (0=identical); convert to similarity
                    "similarity": 1 - distances[i],
                }
            )
        return books

    def get_all_books(self) -> list:
        """
        Fetch every book in the collection with its metadata, for structured
        (non-semantic) filtering — e.g. by publication year, genre, or author.
        """
        try:
            results = self.collection.get(include=["metadatas", "documents"])
        except Exception as exc:
            raise VectorStoreException(f"ChromaDB get failed: {exc}") from exc

        ids = results.get("ids") or []
        documents = results.get("documents") or []
        metadatas = results.get("metadatas") or []

        return [
            {"id": ids[i], "document": documents[i], "metadata": metadatas[i]}
            for i in range(len(ids))
        ]
