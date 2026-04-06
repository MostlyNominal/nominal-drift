"""
nominal_drift.llm
==============
LLM reasoning layer — Tier A of the NominalDrift three-tier architecture.

Responsibilities
----------------
  - Parse user intent from free-form scientific queries
  - Extract structured parameters (composition, HT schedule, element names)
    from unstructured natural-language input
  - Generate scientifically accurate, assumption-aware explanations of
    deterministic engine outputs
  - Produce technical report sections in formal engineering language
  - Flag ambiguities and request missing parameters before calculation

Hard constraint
---------------
The LLM layer must NOT compute numerical results.  All numbers originate
in nominal_drift.science.  The LLM only narrates.

Planned modules
---------------
client      : Ollama REST API wrapper (Day 2)
extractors  : Composition / HT schedule extraction from free text (Day 8)
prompts/    : Jinja2 prompt templates (Day 8)
"""
