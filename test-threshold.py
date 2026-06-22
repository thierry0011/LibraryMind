import sys

sys.path.insert(0, ".")
from app.services.embedding_service import EmbeddingsService
from app.infrastructure.vector_store import VectorStore

es = EmbeddingsService()
vs = VectorStore()

query_embedding = es.embed("desert planets science fiction")
results = vs.search_books(query_embedding, top_k=5)

for r in results:
    print(f"{r['metadata']['title']} — similarity: {r['similarity']:.4f}")
