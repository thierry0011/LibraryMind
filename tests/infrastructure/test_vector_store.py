"""
Tests for infrastructure.vector_store.VectorStore.

chromadb.PersistentClient is mocked throughout — no filesystem access occurs.

chromadb 0.4.24 references np.float_ which was removed in NumPy 2.0, so the
real package cannot be imported. We stub the entire module in sys.modules
before importing VectorStore so the incompatible C-extension is never touched.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

# Inject stub BEFORE the project import pulls in the real (broken) chromadb.
sys.modules.setdefault("chromadb", MagicMock())

from infrastructure.vector_store import VectorStore  # noqa: E402


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def store_setup():
    """
    Yields (VectorStore, mock_collection, mock_client, MockPersistentClient).
    The chromadb patch stays active for the lifetime of each test.
    """
    mock_collection = MagicMock()
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection

    with patch("infrastructure.vector_store.chromadb.PersistentClient") as MockClient:
        MockClient.return_value = mock_client
        vs = VectorStore()
        yield vs, mock_collection, mock_client, MockClient


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


class TestVectorStoreInit:
    def test_persistent_client_called_once(self, store_setup) -> None:
        _, _, _, MockClient = store_setup
        MockClient.assert_called_once()

    def test_persistent_client_receives_correct_path(self, store_setup) -> None:
        _, _, _, MockClient = store_setup
        MockClient.assert_called_once_with(path="./books_chroma_db")

    def test_get_or_create_collection_called_once(self, store_setup) -> None:
        _, _, mock_client, _ = store_setup
        mock_client.get_or_create_collection.assert_called_once()

    def test_collection_name_is_books(self, store_setup) -> None:
        _, _, mock_client, _ = store_setup
        kwargs = mock_client.get_or_create_collection.call_args[1]
        assert kwargs["name"] == "books"

    def test_collection_uses_cosine_space(self, store_setup) -> None:
        _, _, mock_client, _ = store_setup
        kwargs = mock_client.get_or_create_collection.call_args[1]
        assert kwargs["metadata"]["hnsw:space"] == "cosine"

    def test_collection_ef_construction(self, store_setup) -> None:
        _, _, mock_client, _ = store_setup
        kwargs = mock_client.get_or_create_collection.call_args[1]
        assert kwargs["metadata"]["hnsw:ef_construction"] == 200

    def test_collection_M_value(self, store_setup) -> None:
        _, _, mock_client, _ = store_setup
        kwargs = mock_client.get_or_create_collection.call_args[1]
        assert kwargs["metadata"]["hnsw:M"] == 16

    def test_client_attribute_set(self, store_setup) -> None:
        vs, _, mock_client, _ = store_setup
        assert vs.client is mock_client

    def test_collection_attribute_set(self, store_setup) -> None:
        vs, mock_collection, _, _ = store_setup
        assert vs.collection is mock_collection


# ---------------------------------------------------------------------------
# upsert_books
# ---------------------------------------------------------------------------


class TestUpsertBooks:
    def test_upsert_called_once(self, store_setup) -> None:
        vs, mock_collection, _, _ = store_setup
        vs.upsert_books("id-1", [0.1, 0.2], {"title": "Dune"}, "Dune text")
        mock_collection.upsert.assert_called_once()

    def test_id_wrapped_in_list(self, store_setup) -> None:
        vs, mock_collection, _, _ = store_setup
        vs.upsert_books("book-99", [0.1], {}, "doc")
        kwargs = mock_collection.upsert.call_args[1]
        assert kwargs["ids"] == ["book-99"]

    def test_embedding_wrapped_in_list(self, store_setup) -> None:
        vs, mock_collection, _, _ = store_setup
        emb = [0.1, 0.2, 0.3]
        vs.upsert_books("id", emb, {}, "doc")
        kwargs = mock_collection.upsert.call_args[1]
        assert kwargs["embeddings"] == [emb]

    def test_metadata_wrapped_in_list(self, store_setup) -> None:
        vs, mock_collection, _, _ = store_setup
        meta = {"author": "Herbert", "year": 1965}
        vs.upsert_books("id", [0.1], meta, "doc")
        kwargs = mock_collection.upsert.call_args[1]
        assert kwargs["metadatas"] == [meta]

    def test_document_wrapped_in_list(self, store_setup) -> None:
        vs, mock_collection, _, _ = store_setup
        vs.upsert_books("id", [0.1], {}, "full book text")
        kwargs = mock_collection.upsert.call_args[1]
        assert kwargs["documents"] == ["full book text"]

    def test_multiple_upserts_each_call_collection(self, store_setup) -> None:
        vs, mock_collection, _, _ = store_setup
        vs.upsert_books("id-1", [0.1], {}, "doc1")
        vs.upsert_books("id-2", [0.2], {}, "doc2")
        assert mock_collection.upsert.call_count == 2

    def test_upsert_passes_all_four_kwargs(self, store_setup) -> None:
        vs, mock_collection, _, _ = store_setup
        vs.upsert_books("id", [0.5], {"k": "v"}, "text")
        kwargs = mock_collection.upsert.call_args[1]
        assert set(kwargs.keys()) == {"ids", "embeddings", "metadatas", "documents"}


# ---------------------------------------------------------------------------
# search_books — query forwarding
# ---------------------------------------------------------------------------


class TestSearchBooksQuery:
    def test_query_called_once(self, store_setup) -> None:
        vs, mock_collection, _, _ = store_setup
        mock_collection.query.return_value = {
            "ids": [],
            "documents": [],
            "metadatas": [],
            "distances": [],
        }
        vs.search_books([0.1, 0.2])
        mock_collection.query.assert_called_once()

    def test_query_embedding_wrapped_in_list(self, store_setup) -> None:
        vs, mock_collection, _, _ = store_setup
        mock_collection.query.return_value = {
            "ids": [],
            "documents": [],
            "metadatas": [],
            "distances": [],
        }
        emb = [0.1, 0.2, 0.3]
        vs.search_books(emb)
        kwargs = mock_collection.query.call_args[1]
        assert kwargs["query_embeddings"] == [emb]

    def test_default_top_k_is_5(self, store_setup) -> None:
        vs, mock_collection, _, _ = store_setup
        mock_collection.query.return_value = {
            "ids": [],
            "documents": [],
            "metadatas": [],
            "distances": [],
        }
        vs.search_books([0.1])
        kwargs = mock_collection.query.call_args[1]
        assert kwargs["n_results"] == 5

    def test_custom_top_k_passed_through(self, store_setup) -> None:
        vs, mock_collection, _, _ = store_setup
        mock_collection.query.return_value = {
            "ids": [],
            "documents": [],
            "metadatas": [],
            "distances": [],
        }
        vs.search_books([0.1], top_k=10)
        kwargs = mock_collection.query.call_args[1]
        assert kwargs["n_results"] == 10

    def test_include_fields_are_correct(self, store_setup) -> None:
        vs, mock_collection, _, _ = store_setup
        mock_collection.query.return_value = {
            "ids": [],
            "documents": [],
            "metadatas": [],
            "distances": [],
        }
        vs.search_books([0.1])
        kwargs = mock_collection.query.call_args[1]
        assert set(kwargs["include"]) == {"metadatas", "documents", "distances"}


# ---------------------------------------------------------------------------
# search_books — result mapping
# ---------------------------------------------------------------------------


def _make_query_result(ids, documents, metadatas, distances):
    """Build a chromadb-style query result (outer list per query)."""
    return {
        "ids": [ids] if ids is not None else [],
        "documents": [documents] if documents is not None else [],
        "metadatas": [metadatas] if metadatas is not None else [],
        "distances": [distances] if distances is not None else [],
    }


class TestSearchBooksResultMapping:
    def test_empty_results_returns_empty_list(self, store_setup) -> None:
        vs, mock_collection, _, _ = store_setup
        mock_collection.query.return_value = {
            "ids": [],
            "documents": [],
            "metadatas": [],
            "distances": [],
        }
        assert vs.search_books([0.1]) == []

    def test_single_result_returns_one_book(self, store_setup) -> None:
        vs, mock_collection, _, _ = store_setup
        mock_collection.query.return_value = _make_query_result(
            ["book-1"], ["Dune text"], [{"author": "Herbert"}], [0.1]
        )
        results = vs.search_books([0.1])
        assert len(results) == 1

    def test_single_result_id_mapped(self, store_setup) -> None:
        vs, mock_collection, _, _ = store_setup
        mock_collection.query.return_value = _make_query_result(
            ["book-42"], ["text"], [{}], [0.0]
        )
        assert vs.search_books([0.1])[0]["id"] == "book-42"

    def test_single_result_document_mapped(self, store_setup) -> None:
        vs, mock_collection, _, _ = store_setup
        mock_collection.query.return_value = _make_query_result(
            ["id"], ["The full document text"], [{}], [0.0]
        )
        assert vs.search_books([0.1])[0]["document"] == "The full document text"

    def test_single_result_metadata_mapped(self, store_setup) -> None:
        vs, mock_collection, _, _ = store_setup
        meta = {"title": "Dune", "genre": "sci-fi"}
        mock_collection.query.return_value = _make_query_result(
            ["id"], ["doc"], [meta], [0.0]
        )
        assert vs.search_books([0.1])[0]["metadata"] == meta

    def test_similarity_is_one_minus_distance(self, store_setup) -> None:
        vs, mock_collection, _, _ = store_setup
        mock_collection.query.return_value = _make_query_result(
            ["id"], ["doc"], [{}], [0.3]
        )
        result = vs.search_books([0.1])[0]
        assert result["similarity"] == pytest.approx(0.7)

    def test_similarity_zero_distance_gives_one(self, store_setup) -> None:
        vs, mock_collection, _, _ = store_setup
        mock_collection.query.return_value = _make_query_result(
            ["id"], ["doc"], [{}], [0.0]
        )
        assert vs.search_books([0.1])[0]["similarity"] == pytest.approx(1.0)

    def test_similarity_max_distance_gives_zero(self, store_setup) -> None:
        vs, mock_collection, _, _ = store_setup
        mock_collection.query.return_value = _make_query_result(
            ["id"], ["doc"], [{}], [1.0]
        )
        assert vs.search_books([0.1])[0]["similarity"] == pytest.approx(0.0)

    def test_multiple_results_all_returned(self, store_setup) -> None:
        vs, mock_collection, _, _ = store_setup
        mock_collection.query.return_value = _make_query_result(
            ["b1", "b2", "b3"],
            ["doc1", "doc2", "doc3"],
            [{}, {}, {}],
            [0.1, 0.2, 0.3],
        )
        assert len(vs.search_books([0.1])) == 3

    def test_multiple_results_ids_in_order(self, store_setup) -> None:
        vs, mock_collection, _, _ = store_setup
        mock_collection.query.return_value = _make_query_result(
            ["first", "second"], ["d1", "d2"], [{}, {}], [0.1, 0.5]
        )
        results = vs.search_books([0.1])
        assert results[0]["id"] == "first"
        assert results[1]["id"] == "second"

    def test_multiple_results_similarities_in_order(self, store_setup) -> None:
        vs, mock_collection, _, _ = store_setup
        mock_collection.query.return_value = _make_query_result(
            ["a", "b"], ["d1", "d2"], [{}, {}], [0.2, 0.6]
        )
        results = vs.search_books([0.1])
        assert results[0]["similarity"] == pytest.approx(0.8)
        assert results[1]["similarity"] == pytest.approx(0.4)

    def test_result_dict_has_exactly_four_keys(self, store_setup) -> None:
        vs, mock_collection, _, _ = store_setup
        mock_collection.query.return_value = _make_query_result(
            ["id"], ["doc"], [{}], [0.1]
        )
        result = vs.search_books([0.1])[0]
        assert set(result.keys()) == {"id", "document", "metadata", "similarity"}

    def test_returns_list_type(self, store_setup) -> None:
        vs, mock_collection, _, _ = store_setup
        mock_collection.query.return_value = {
            "ids": [],
            "documents": [],
            "metadatas": [],
            "distances": [],
        }
        assert isinstance(vs.search_books([0.1]), list)
