"""
nominal_drift.knowledge
====================
Local experimental knowledge layer — retrieval-based continuous learning
without LLM retraining.

Architecture:
  - Structured experiment store (SQLite) for exact-match retrieval
  - Semantic similarity store (ChromaDB / LanceDB) for fuzzy search
  - File attachment index linking experiment IDs to raw data files

The knowledge layer is private to each user installation.  It is never
committed to the public repository (enforced via .gitignore on data/).

Planned modules
---------------
schema_db         : SQLAlchemy ORM model for the experiment table
experiment_store  : CRUD interface (write / read / list / query)
similarity_search : Embedding-based retrieval (Phase 1b)
comparison_engine : Delta calculation between stored experiments (Phase 2)
"""
