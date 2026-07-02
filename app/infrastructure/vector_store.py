import chromadb
from pathlib import Path

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
        self.collection.upsert(
            ids=[id], embeddings=[embedding], metadatas=[metadata], documents=[document]
        )

    def search_books(self, query_embedding: list, top_k: int = 5) -> list:
        """
        Search for books in the vector store.
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["metadatas", "documents", "distances"],
        )
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
                    "similarity": 1 - distances[i],
                }
            )
        return books
