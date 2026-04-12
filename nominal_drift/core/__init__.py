"""
nominal_drift.core
===============
Orchestration layer — Tier C of the NominalDrift three-tier architecture.

Responsibilities
----------------
  - Route user requests to the correct scientific module(s)
  - Validate structured inputs against Pydantic schemas
  - Manage session state (loaded composition, active HT schedule)
  - Cache expensive simulation results within a session
  - Handle typed exceptions and format error messages for Tier A
  - Enforce module versioning and compatibility checks

Planned modules
---------------
orchestrator  : Main request router (Day 9)
session       : Session context dataclass (Day 9)
registry      : Module registry — name → callable + schema (Phase 2)
exceptions    : Typed exception hierarchy shared across all tiers
"""
