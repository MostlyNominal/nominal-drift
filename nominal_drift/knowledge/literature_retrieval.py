"""
nominal_drift.knowledge.literature_retrieval
=============================================
Keyword-based retrieval over a LiteratureStore.

PURPOSE: Simple TF-based retrieval to find relevant documents without vector embeddings.
Tokenises queries, counts term hits, and returns ranked results with relevance scores.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from nominal_drift.knowledge.literature_store import LiteratureDocument, LiteratureStore


# ---------------------------------------------------------------------------
# Result Schema
# ---------------------------------------------------------------------------

class RetrievalResult(BaseModel, frozen=True):
    """Result of a single document retrieval.

    Attributes
    ----------
    doc_id : str
        Document ID.
    title : str
        Document title.
    relevance_score : float
        Score in [0, 1] — number of matched query terms / total query terms.
    matched_terms : list[str]
        Query terms that matched in this document.
    excerpt : str
        First 200 chars of content containing a matched term.
    """

    doc_id: str
    title: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    matched_terms: list[str]
    excerpt: str


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    store: LiteratureStore,
    top_k: int = 5,
    min_score: float = 0.0,
) -> list[RetrievalResult]:
    """Perform keyword-based retrieval on a literature store.

    Algorithm:
      1. Tokenise query by whitespace, convert to lowercase.
      2. For each document, count unique query terms in title + content
         (case-insensitive matching).
      3. relevance_score = n_matched / n_query_terms (clamped to [0, 1]).
      4. Extract excerpt: first 200 chars of content containing the first matched term.
      5. Return top_k results by score, filtering by min_score.

    Parameters
    ----------
    query : str
        Search query (whitespace-tokenised).
    store : LiteratureStore
        Literature store to search.
    top_k : int
        Maximum number of results to return (default 5).
    min_score : float
        Minimum relevance score threshold [0, 1] (default 0.0).

    Returns
    -------
    list[RetrievalResult]
        Ranked retrieval results, highest score first.
    """
    # --------- Tokenise query ---------
    query_terms = [term.lower().strip() for term in query.split()]
    query_terms = [t for t in query_terms if t]  # remove empty strings
    n_query_terms = max(len(query_terms), 1)  # avoid division by zero

    if not query_terms:
        return []

    # --------- Score each document ---------
    scored: list[tuple[float, RetrievalResult]] = []

    for doc in store.list_documents():
        # Combine title and content for searching
        searchable_text = (doc.title + " " + doc.content).lower()

        # Count matched terms
        matched_terms: list[str] = []
        for term in query_terms:
            if term in searchable_text:
                matched_terms.append(term)

        # Compute relevance score
        relevance_score = len(matched_terms) / n_query_terms

        # Filter by minimum score
        if relevance_score < min_score:
            continue

        # Extract excerpt: first 200 chars containing first matched term
        excerpt = ""
        if matched_terms:
            first_match = matched_terms[0]
            idx = searchable_text.find(first_match)
            if idx >= 0:
                start = max(0, idx - 50)
                end = min(len(searchable_text), start + 200)
                excerpt = searchable_text[start:end].strip()

        result = RetrievalResult(
            doc_id=doc.doc_id,
            title=doc.title,
            relevance_score=relevance_score,
            matched_terms=matched_terms,
            excerpt=excerpt,
        )

        scored.append((relevance_score, result))

    # --------- Sort and return top-k ---------
    scored.sort(key=lambda x: x[0], reverse=True)
    return [result for _, result in scored[:top_k]]
