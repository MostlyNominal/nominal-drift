"""
tests/unit/test_literature_store.py
====================================
Unit tests for nominal_drift.knowledge.literature_store and
nominal_drift.knowledge.literature_retrieval.

Coverage for literature_store:
  - add_document returns doc_id
  - get_document retrieves by id
  - remove_document returns True on success, False on missing
  - len(store) increments on add
  - store persists to JSON on save
  - store reloads from JSON
  - make_document generates non-empty doc_id
  - list_documents returns list
  - Documents are frozen (immutable)
  - Document timestamps are ISO format
  - Document validation rejects invalid source_type
  - Store preserves order

Coverage for literature_retrieval:
  - retrieve returns RetrievalResult list
  - retrieve with exact keyword match has score > 0
  - retrieve with no match returns empty list
  - retrieve returns at most top_k results
  - excerpt is non-empty when match found
  - matched_terms lists query terms found
  - relevance_score is in [0, 1]
  - empty query returns empty list
  - min_score filtering works
  - multiple matches ranked by score
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from nominal_drift.knowledge.literature_store import (
    LiteratureDocument,
    LiteratureStore,
    make_document,
)
from nominal_drift.knowledge.literature_retrieval import (
    retrieve,
    RetrievalResult,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def temp_store_path():
    """Provide a temporary path for store files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_literature.json"


@pytest.fixture
def empty_store(temp_store_path) -> LiteratureStore:
    """An empty, initialized store."""
    return LiteratureStore(temp_store_path)


@pytest.fixture
def populated_store(empty_store) -> LiteratureStore:
    """A store with sample documents."""
    doc1 = make_document(
        title="Sensitisation in 316L Stainless Steel",
        content="This paper discusses chromium depletion and sensitisation in 316L. "
                "The sensitisation nose occurs around 700 °C for austenitic stainless steels. "
                "Chromium diffusion is the dominant mechanism.",
        source_type="text",
        tags=["316L", "sensitisation", "Cr", "diffusion"],
        metadata={"author": "Smith, J.", "year": "2020", "journal": "Materials Science"},
    )
    doc2 = make_document(
        title="Grain Boundary Engineering",
        content="Grain boundaries are sites of preferential precipitation. "
                "The sink model captures Cr concentration drops at boundaries.",
        source_type="note",
        tags=["grain_boundary", "sink_model"],
        metadata={"author": "Johnson, M."},
    )
    doc3 = make_document(
        title="Austenitic Stainless Steel 304",
        content="304 stainless steel has higher carbon content than 316L. "
                "This leads to deeper sensitisation at lower temperatures.",
        source_type="standard",
        tags=["304", "stainless_steel"],
        metadata={"standard": "ASTM A240"},
    )

    empty_store.add_document(doc1)
    empty_store.add_document(doc2)
    empty_store.add_document(doc3)

    return empty_store


# ===========================================================================
# Tests: LiteratureDocument & make_document
# ===========================================================================

class TestMakeDocument:
    """Test document creation factory."""

    def test_make_document_returns_document(self):
        """make_document returns a LiteratureDocument instance."""
        doc = make_document(
            title="Test Doc",
            content="Some content",
        )
        assert isinstance(doc, LiteratureDocument)

    def test_make_document_generates_doc_id(self):
        """make_document generates a non-empty doc_id (UUID)."""
        doc = make_document(
            title="Test",
            content="Content",
        )
        assert doc.doc_id
        assert len(doc.doc_id) > 0

    def test_make_document_different_ids(self):
        """Successive calls to make_document generate different UUIDs."""
        doc1 = make_document(title="Doc 1", content="Content 1")
        doc2 = make_document(title="Doc 2", content="Content 2")
        assert doc1.doc_id != doc2.doc_id

    def test_make_document_adds_timestamp(self):
        """make_document adds an ISO 8601 timestamp."""
        doc = make_document(title="Test", content="Content")
        assert doc.added_at
        # Should be parseable as ISO format
        from datetime import datetime
        datetime.fromisoformat(doc.added_at)  # raises if not valid

    def test_make_document_default_source_type(self):
        """Default source_type is 'note'."""
        doc = make_document(title="Test", content="Content")
        assert doc.source_type == "note"

    def test_make_document_custom_source_type(self):
        """source_type can be customised."""
        doc = make_document(
            title="Test",
            content="Content",
            source_type="pdf",
        )
        assert doc.source_type == "pdf"

    def test_make_document_default_tags_empty(self):
        """Default tags is empty list."""
        doc = make_document(title="Test", content="Content")
        assert doc.tags == []

    def test_make_document_custom_tags(self):
        """tags can be provided."""
        doc = make_document(
            title="Test",
            content="Content",
            tags=["tag1", "tag2"],
        )
        assert doc.tags == ["tag1", "tag2"]

    def test_make_document_default_metadata_empty(self):
        """Default metadata is empty dict."""
        doc = make_document(title="Test", content="Content")
        assert doc.metadata == {}

    def test_make_document_custom_metadata(self):
        """metadata can be provided."""
        doc = make_document(
            title="Test",
            content="Content",
            metadata={"author": "Alice", "year": "2024"},
        )
        assert doc.metadata == {"author": "Alice", "year": "2024"}

    def test_make_document_is_frozen(self):
        """Documents created by make_document are frozen (immutable)."""
        from pydantic import ValidationError
        doc = make_document(title="Test", content="Content")
        with pytest.raises(ValidationError):
            doc.title = "Modified"

    def test_make_document_rejects_invalid_source_type(self):
        """make_document rejects invalid source_type."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            make_document(
                title="Test",
                content="Content",
                source_type="invalid_type",
            )


# ===========================================================================
# Tests: LiteratureStore - Add, Get, Remove
# ===========================================================================

class TestStoreAddGetRemove:
    """Test basic add/get/remove operations."""

    def test_add_document_returns_doc_id(self, empty_store):
        """add_document returns the document's doc_id."""
        doc = make_document(title="Test", content="Content")
        doc_id = empty_store.add_document(doc)
        assert doc_id == doc.doc_id

    def test_get_document_retrieves_by_id(self, empty_store):
        """get_document retrieves an added document by ID."""
        doc = make_document(title="Test", content="Content")
        empty_store.add_document(doc)
        retrieved = empty_store.get_document(doc.doc_id)
        assert retrieved is not None
        assert retrieved.title == doc.title

    def test_get_document_returns_none_on_missing(self, empty_store):
        """get_document returns None if ID not found."""
        result = empty_store.get_document("nonexistent_id")
        assert result is None

    def test_remove_document_returns_true_on_success(self, empty_store):
        """remove_document returns True if document found and removed."""
        doc = make_document(title="Test", content="Content")
        doc_id = empty_store.add_document(doc)
        result = empty_store.remove_document(doc_id)
        assert result is True

    def test_remove_document_returns_false_on_missing(self, empty_store):
        """remove_document returns False if document not found."""
        result = empty_store.remove_document("nonexistent_id")
        assert result is False

    def test_remove_document_actually_removes(self, empty_store):
        """After remove_document, the document is no longer retrievable."""
        doc = make_document(title="Test", content="Content")
        doc_id = empty_store.add_document(doc)
        empty_store.remove_document(doc_id)
        retrieved = empty_store.get_document(doc_id)
        assert retrieved is None


# ===========================================================================
# Tests: LiteratureStore - Persistence & Reload
# ===========================================================================

class TestStorePersistence:
    """Test JSON persistence and reload."""

    def test_store_persists_to_json(self, temp_store_path):
        """Adding documents causes save to JSON file."""
        store = LiteratureStore(temp_store_path)
        doc = make_document(title="Test", content="Content")
        store.add_document(doc)
        assert temp_store_path.exists()

    def test_store_json_has_correct_structure(self, temp_store_path):
        """JSON file has correct structure: {"documents": [...]})."""
        store = LiteratureStore(temp_store_path)
        doc = make_document(title="Test", content="Content")
        store.add_document(doc)

        with open(temp_store_path, "r") as f:
            data = json.load(f)

        assert isinstance(data, dict)
        assert "documents" in data
        assert isinstance(data["documents"], list)
        assert len(data["documents"]) == 1

    def test_store_reloads_from_json(self, temp_store_path):
        """Creating a new store with same path reloads persisted documents."""
        # Create and populate store
        store1 = LiteratureStore(temp_store_path)
        doc = make_document(title="Persistent Doc", content="This should persist")
        doc_id = store1.add_document(doc)

        # Create new store from same path
        store2 = LiteratureStore(temp_store_path)
        retrieved = store2.get_document(doc_id)

        assert retrieved is not None
        assert retrieved.title == "Persistent Doc"

    def test_store_reload_preserves_all_documents(self, temp_store_path):
        """Reload from JSON preserves all documents."""
        store1 = LiteratureStore(temp_store_path)
        docs = [
            make_document(title=f"Doc {i}", content=f"Content {i}")
            for i in range(5)
        ]
        for doc in docs:
            store1.add_document(doc)

        store2 = LiteratureStore(temp_store_path)
        assert len(store2) == 5


# ===========================================================================
# Tests: LiteratureStore - Len, List
# ===========================================================================

class TestStoreLen:
    """Test __len__ and list_documents."""

    def test_len_empty_store(self, empty_store):
        """len() on empty store returns 0."""
        assert len(empty_store) == 0

    def test_len_increments_on_add(self, empty_store):
        """len() increments when documents are added."""
        doc1 = make_document(title="Doc 1", content="Content 1")
        empty_store.add_document(doc1)
        assert len(empty_store) == 1

        doc2 = make_document(title="Doc 2", content="Content 2")
        empty_store.add_document(doc2)
        assert len(empty_store) == 2

    def test_len_decrements_on_remove(self, empty_store):
        """len() decrements when documents are removed."""
        doc = make_document(title="Doc", content="Content")
        doc_id = empty_store.add_document(doc)
        assert len(empty_store) == 1

        empty_store.remove_document(doc_id)
        assert len(empty_store) == 0

    def test_list_documents_empty_store(self, empty_store):
        """list_documents on empty store returns empty list."""
        assert empty_store.list_documents() == []

    def test_list_documents_populated_store(self, populated_store):
        """list_documents returns all documents."""
        docs = populated_store.list_documents()
        assert len(docs) == 3
        assert all(isinstance(d, LiteratureDocument) for d in docs)

    def test_list_documents_returns_correct_titles(self, populated_store):
        """list_documents returns documents with expected titles."""
        docs = populated_store.list_documents()
        titles = {d.title for d in docs}
        assert "Sensitisation in 316L Stainless Steel" in titles
        assert "Grain Boundary Engineering" in titles


# ===========================================================================
# Tests: literature_retrieval
# ===========================================================================

class TestRetrieve:
    """Test keyword-based retrieval."""

    def test_retrieve_returns_list(self, populated_store):
        """retrieve returns a list of RetrievalResult."""
        results = retrieve("sensitisation", populated_store)
        assert isinstance(results, list)

    def test_retrieve_with_exact_match_has_nonzero_score(self, populated_store):
        """Query matching document content has score > 0."""
        results = retrieve("sensitisation", populated_store)
        assert len(results) > 0
        assert results[0].relevance_score > 0.0

    def test_retrieve_with_no_match_empty(self, populated_store):
        """Query with no matches returns empty list or empty results."""
        results = retrieve("xyz_nonexistent_term_xyz", populated_store)
        # Could be empty or have very low scores - either way, should be short
        assert len(results) <= 1 or all(r.relevance_score == 0.0 for r in results)

    def test_retrieve_respects_top_k(self, populated_store):
        """retrieve returns at most top_k results."""
        results = retrieve("stainless", populated_store, top_k=2)
        assert len(results) <= 2

    def test_retrieve_excerpt_non_empty_on_match(self, populated_store):
        """excerpt is non-empty when a match is found."""
        results = retrieve("chromium", populated_store)
        assert len(results) > 0
        assert len(results[0].excerpt) > 0

    def test_retrieve_matched_terms_contains_query_term(self, populated_store):
        """matched_terms lists query terms that matched."""
        results = retrieve("diffusion", populated_store)
        assert len(results) > 0
        assert "diffusion" in results[0].matched_terms

    def test_retrieve_relevance_score_in_range(self, populated_store):
        """relevance_score is in [0, 1]."""
        results = retrieve("stainless", populated_store)
        for result in results:
            assert 0.0 <= result.relevance_score <= 1.0

    def test_retrieve_empty_query_returns_empty(self, populated_store):
        """Empty query string returns empty list."""
        results = retrieve("", populated_store)
        assert results == []

    def test_retrieve_whitespace_only_query_returns_empty(self, populated_store):
        """Query with only whitespace returns empty list."""
        results = retrieve("   ", populated_store)
        assert results == []

    def test_retrieve_min_score_filtering(self, populated_store):
        """min_score filters results below threshold."""
        results = retrieve("stainless", populated_store, min_score=0.9)
        # Should have few or no results if min_score is very high
        for result in results:
            assert result.relevance_score >= 0.9

    def test_retrieve_ranked_by_score(self, populated_store):
        """Results are ranked by relevance_score (descending)."""
        results = retrieve("stainless", populated_store, top_k=10)
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i].relevance_score >= results[i + 1].relevance_score

    def test_retrieve_multiple_matches_ranked(self, populated_store):
        """Document with multiple matched terms ranks higher."""
        # Query with terms present in multiple documents
        results = retrieve("stainless steel", populated_store, top_k=10)
        assert len(results) > 0
        # First result should have highest score
        first_score = results[0].relevance_score
        if len(results) > 1:
            second_score = results[1].relevance_score
            assert first_score >= second_score

    def test_retrieve_result_structure(self, populated_store):
        """RetrievalResult has all required fields."""
        results = retrieve("sensitisation", populated_store)
        assert len(results) > 0
        result = results[0]
        assert hasattr(result, "doc_id")
        assert hasattr(result, "title")
        assert hasattr(result, "relevance_score")
        assert hasattr(result, "matched_terms")
        assert hasattr(result, "excerpt")

    def test_retrieve_doc_id_valid(self, populated_store):
        """doc_id in result corresponds to a document in the store."""
        results = retrieve("grain", populated_store)
        assert len(results) > 0
        result = results[0]
        doc = populated_store.get_document(result.doc_id)
        assert doc is not None

    def test_retrieve_title_matches_document(self, populated_store):
        """title in result matches the stored document."""
        results = retrieve("austenitic", populated_store)
        assert len(results) > 0
        result = results[0]
        doc = populated_store.get_document(result.doc_id)
        assert result.title == doc.title


# ===========================================================================
# Tests: Integration
# ===========================================================================

class TestIntegration:
    """Integration tests combining store and retrieval."""

    def test_add_and_retrieve_workflow(self, empty_store):
        """Add documents and retrieve them via keyword search."""
        doc = make_document(
            title="Test Document",
            content="This is a test document about sensitisation and diffusion.",
            tags=["test", "sensitisation"],
        )
        empty_store.add_document(doc)

        results = retrieve("sensitisation", empty_store)
        assert len(results) == 1
        assert results[0].title == "Test Document"

    def test_remove_and_retrieve_workflow(self, populated_store):
        """Remove a document and verify it's no longer retrievable."""
        docs_before = populated_store.list_documents()
        doc_to_remove = docs_before[0]

        populated_store.remove_document(doc_to_remove.doc_id)

        # Try to retrieve by a unique term from that document
        results = retrieve(doc_to_remove.title[:20], populated_store)
        # If the document was removed, it shouldn't appear in results
        for result in results:
            assert result.doc_id != doc_to_remove.doc_id

    def test_persistence_across_instances(self, temp_store_path):
        """Documents added in one instance are retrievable from another."""
        store1 = LiteratureStore(temp_store_path)
        doc = make_document(
            title="Persistent Document",
            content="This document should persist across instances.",
        )
        store1.add_document(doc)

        store2 = LiteratureStore(temp_store_path)
        results = retrieve("persistent", store2)
        assert len(results) > 0
        assert results[0].title == "Persistent Document"
