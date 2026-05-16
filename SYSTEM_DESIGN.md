# System Design: Adaptive Conversation Intelligence — Sync Architecture

## Overview

Personal data stays local by default. Only derived, non-sensitive artefacts sync to cloud. Raw conversations never leave the device.

## Architecture Diagram
┌─────────────────────────────────────────────┐
│                USER DEVICE                  │
│                                             │
│  Raw Conversations ──▶ Processing Pipeline  │
│  (never synced)         • Drift detection   │
│                         • Intent classify   │
│                         • Topic indexing    │
│                         • Conflict resolve  │
│                              │              │
│                              ▼              │
│                    On-Device Store          │
│                    • persona.json           │
│                    • drift_report.json      │
│                    • faiss_index.pkl        │
│                    • intent_classifier.pkl  │
└──────────────────────┬──────────────────────┘
│ sync (derived only)
▼
┌────────────────────────┐
│      CLOUD LAYER       │
│ • persona snapshot     │
│ • drift timeline       │
│ • intent model weights │
│ NOT stored:            │
│ • raw conversations    │
│ • faiss index          │
└────────────────────────┘

## What Syncs vs Stays Local

| Artefact | Size | Syncs? |
|---|---|---|
| Raw conversations | ~50MB | ❌ Never |
| FAISS index | ~150MB | ❌ Too large |
| persona.json | ~20KB | ✅ Yes |
| drift_report.json | ~500KB | ✅ Yes |
| intent_classifier.pkl | ~150KB | ✅ Yes |
| topic_checkpoints.pkl | ~40MB | ❌ Local only |

## Conflict Resolution

Last-write-wins with field-level merge for persona. Drift reports append chronologically. Intent model: higher F1 wins.

## Failure Recovery

| Failure | Recovery |
|---|---|
| Cloud unavailable | Full offline mode continues |
| FAISS corrupted | Rebuild from checkpoints (~10 min) |
| Persona missing | Re-run persona_extractor.py |
| Intent model missing | Retrain in ~2 seconds |