"""
nominal_drift.knowledge.literature_store
==========================================
Local-first JSON-backed literature document store.

PURPOSE: Store and manage literature documents (PDFs, text, notes, standards, excerpts)
with no cloud dependencies or vector embeddings.

Data model:
  - LiteratureDocument: frozen Pydantic v2 model with UUID, title, content, tags, metadata
  - LiteratureStore: JSON-file-backed store with add/get/list/remove operations
  - make_document(): factory for creating documents with auto-generated UUID and timestamp
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DocumentType = Literal["pdf", "text", "note", "standard", "excerpt"]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class LiteratureDocument(BaseModel):
    """A frozen literature document in the store.

    Attributes
    ----------
    doc_id : str
        Unique document identifier (UUID4).
    title : str
        Document title or descriptive heading.
    source_type : str
        Document type: "pdf", "text", "note", "standard", "excerpt".
    content : str
        Extracted or full text content.
    filepath : str | None
        Original file path if imported from disk.
    tags : list[str]
        Free-form tags for categorisation (e.g. ["316L", "sensitisation"]).
    added_at : str
        ISO 8601 UTC timestamp when added to the store.
    metadata : dict[str, str]
        Key-value metadata (e.g. author, year, journal, doi).
    """

    model_config = {"frozen": True}

    doc_id: str = Field(
        ...,
        description="Unique document ID (UUID4).",
        min_length=1,
    )
    title: str = Field(
        ...,
        description="Document title or heading.",
        min_length=1,
    )
    source_type: str = Field(
        ...,
        description='Document type: "pdf", "text", "note", "standard", "excerpt".',
        min_length=1,
    )
    content: str = Field(
        ...,
        description="Extracted or full text content.",
    )
    filepath: str | None = Field(
        default=None,
        description="Original file path if imported from disk.",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Free-form tags for categorisation.",
    )
    added_at: str = Field(
        ...,
        description="ISO 8601 UTC timestamp when added.",
    )
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Key-value metadata (author, year, journal, doi, etc.).",
    )

    @field_validator("source_type")
    @classmethod
    def _validate_source_type(cls, v: str) -> str:
        valid = {"pdf", "text", "note", "standard", "excerpt"}
        if v not in valid:
            raise ValueError(
                f"source_type must be one of {valid}; got {v!r}"
            )
        return v

    @field_validator("tags")
    @classmethod
    def _validate_tags(cls, v: list[str]) -> list[str]:
        for tag in v:
            if not isinstance(tag, str) or not tag.strip():
                raise ValueError("All tags must be non-empty strings.")
        return v


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_document(
    title: str,
    content: str,
    source_type: str = "note",
    filepath: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, str] | None = None,
) -> LiteratureDocument:
    """Create a LiteratureDocument with auto-generated UUID and timestamp.

    Parameters
    ----------
    title : str
        Document title.
    content : str
        Document content (text).
    source_type : str
        Document type (default "note").
    filepath : str | None
        Original file path if applicable.
    tags : list[str] | None
        Free-form tags (default []).
    metadata : dict[str, str] | None
        Key-value metadata (default {}).

    Returns
    -------
    LiteratureDocument
        Frozen document ready for storage.
    """
    return LiteratureDocument(
        doc_id=str(uuid.uuid4()),
        title=title,
        source_type=source_type,
        content=content,
        filepath=filepath,
        tags=tags or [],
        added_at=datetime.now(timezone.utc).isoformat(),
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class LiteratureStore:
    """JSON-file-backed document store.

    A simple, local-first store that persists documents to a JSON file.
    No cloud dependencies or vector embeddings.

    Attributes
    ----------
    store_path : Path
        Path to the JSON store file.
    _documents : dict[str, LiteratureDocument]
        In-memory document registry (doc_id → LiteratureDocument).
    """

    def __init__(self, store_path: str | Path):
        """Initialise the store and load existing documents if the file exists.

        Parameters
        ----------
        store_path : str | Path
            Path where the JSON store file is/will be created.
        """
        self.store_path = Path(store_path)
        self._documents: dict[str, LiteratureDocument] = {}
        self._load()

    def add_document(self, doc: LiteratureDocument) -> str:
        """Add a document to the store and persist to disk.

        Parameters
        ----------
        doc : LiteratureDocument
            Document to add.

        Returns
        -------
        str
            The document's doc_id.
        """
        self._documents[doc.doc_id] = doc
        self.save()
        return doc.doc_id

    def get_document(self, doc_id: str) -> LiteratureDocument | None:
        """Retrieve a document by ID.

        Parameters
        ----------
        doc_id : str
            Document ID.

        Returns
        -------
        LiteratureDocument | None
            The document if found, else None.
        """
        return self._documents.get(doc_id)

    def list_documents(self) -> list[LiteratureDocument]:
        """List all documents in the store.

        Returns
        -------
        list[LiteratureDocument]
            All documents, in insertion order.
        """
        return list(self._documents.values())

    def remove_document(self, doc_id: str) -> bool:
        """Remove a document by ID and persist to disk.

        Parameters
        ----------
        doc_id : str
            Document ID.

        Returns
        -------
        bool
            True if the document was found and removed, else False.
        """
        if doc_id in self._documents:
            del self._documents[doc_id]
            self.save()
            return True
        return False

    def save(self) -> None:
        """Persist all documents to the JSON file."""
        data = {
            "documents": [
                doc.model_dump() for doc in self._documents.values()
            ]
        }
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _load(self) -> None:
        """Load documents from the JSON file if it exists."""
        if not self.store_path.exists():
            return

        with open(self.store_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for doc_data in data.get("documents", []):
            doc = LiteratureDocument(**doc_data)
            self._documents[doc.doc_id] = doc

    def __len__(self) -> int:
        """Return the number of documents in the store."""
        return len(self._documents)
