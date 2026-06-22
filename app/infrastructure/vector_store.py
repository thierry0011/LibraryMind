import chromadb


class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="./books_chroma_db")
        self.collection = self.client.get_or_create_collection(
            name="books",
            metadata={
                "hnsw:space": "cosine",
                "hnsw:ef_construction": 200,
                "hnsw:M": 16,
            },
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
